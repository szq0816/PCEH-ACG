from datetime import datetime
from torch import nn

from hash_model import PCEH
import time

from einops import rearrange
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
import torch.nn.functional as F
from optimization import BertAdam
from utils.calc_utils import calc_neighbor, calc_map_k
from load_data import generate_dataset
from utils.logger import get_logger, clear_logger

dataset_root_path = "/home/lzx/dataSet/original/clip-hash-dataset"


class TrainerAsym:
    """
        train class
    """
    def __init__(self, args):

        self.args = args

        torch.random.manual_seed(seed=self.args.seed)
        torch.autograd.set_detect_anomaly(True)

        os.makedirs(self.args.save_dir, exist_ok=True)
        self._init_writer()

        self.logger.info('Start logging...')

        if self.args.is_train:
            log_str_args = "\n"
            for para in self.args.__dict__:
                log_str_args += " " * (40 - len(para)) + str(para) + "=" + str(self.args.__dict__[para]) + "\n"
            self.logger.info(log_str_args)
        else:
            self.logger.info(f"pretrained: {self.args.pretrained}")

        self.rank = self.args.rank  # gpu rank
        self.bit = self.args.bit
        self.guide_bit = self.args.guide_bit_dim
        self.best_epoch = 0
        self.max_i2t = 0
        self.max_t2i = 0
        self.max_avg = 0

        # buffer
        self.ibuf = {}
        self.tbuf = {}
        self.bbuf = {}

        for one in [self.bit, self.guide_bit]:
            self.ibuf[one] = torch.randn(self.args.train_num, one).to(self.rank, non_blocking=True)
            self.tbuf[one] = torch.randn(self.args.train_num, one).to(self.rank, non_blocking=True)
            self.bbuf[one] = torch.sign(self.ibuf[one] + self.tbuf[one])

        self._init_dataset()
        self._init_model()

        self.device = torch.device("cuda", self.rank)

        self.logger.info("Train dataset len: {}".format(len(self.train_loader.dataset)))

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            self.test()

    def _init_writer(self):

        # logger
        self.logger = get_logger(os.path.join(self.args.save_dir, "train.log" if self.args.is_train else "test.log"))

        with open(os.path.join(self.args.save_dir, "description.txt"), 'w') as f:
            # write description
            f.write("")
            f.close()

    def _init_model(self):
        self.logger.info("init model.")
        self.logger.info("Using ViT & GPT2...")

        self.model = PCEH(args=self.args).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info(f"load pretrained model at {self.args.pretrained}")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()
        self.optimizer = BertAdam(
            [
                {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                {'params': self.model.hash.parameters(), 'lr': self.args.lr},
            ],
            lr=self.args.lr,
            warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)

    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset...")

        global dataset_root_path
        self.args.index_file = os.path.join(dataset_root_path, self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join(dataset_root_path, self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join(dataset_root_path, self.args.dataset, self.args.label_file)

        train_data, query_data, retrieval_data = generate_dataset(captionFile=self.args.caption_file,
                                                                  indexFile=self.args.index_file,
                                                                  labelFile=self.args.label_file,
                                                                  maxWords=self.args.max_words,
                                                                  imageResolution=self.args.resolution,
                                                                  dataset=self.args.dataset,
                                                                  query_num=self.args.query_num,
                                                                  train_num=self.args.train_num,
                                                                  full_ratio=self.args.full_ratio,
                                                                  oimg_ratio=self.args.oimg_ratio,
                                                                  seed=self.args.seed)

        self.query_data = query_data
        self.retrieval_data = retrieval_data

        self.train_labels = train_data.get_all_label().float().to(self.rank, non_blocking=True)
        self.query_labels = query_data.get_all_label().float()
        self.retrieval_labels = retrieval_data.get_all_label().float()
        self.d_m1 = retrieval_data.m1
        self.d_m2 = retrieval_data.m2

        self.args.retrieval_num = len(self.retrieval_labels)

        self.args.num_class = self.query_labels.shape[1]

        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )

    def change_state(self, mode):
        if mode == "train":
            self.model.train()
        elif mode == "valid":
            self.model.eval()

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info("\n\n\n")
        self.logger.info(
            "####################### Train epochs: %d/%d #######################" % (epoch, self.args.epochs))
        epoch_avg_loss_dict = {'all_loss': 0}

        for image, text, key_padding_mask, label, m1, m2, index in tqdm(self.train_loader):

            image = image.float().to(self.rank, non_blocking=True)
            label = label.float().to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)

            m1 = m1.to(self.rank, non_blocking=True)
            m2 = m2.to(self.rank, non_blocking=True)
            image = torch.mul(image, m1.unsqueeze(-1).unsqueeze(-1))
            text = torch.mul(text, m2)

            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)

            output_dict = self.model(image, text, key_padding_mask, m1, m2)

            _B_batch = {}

            img_cls_hash = output_dict['img_cls_hash']
            txt_cls_hash = output_dict['txt_cls_hash']
            self.ibuf[self.bit][index] = img_cls_hash.detach()
            self.tbuf[self.bit][index] = txt_cls_hash.detach()
            _B_batch[self.bit] = self.bbuf[self.bit][index]

            img_cls_guide = output_dict['img_cls_guide']
            txt_cls_guide = output_dict['txt_cls_guide']
            self.ibuf[self.guide_bit][index] = img_cls_guide.detach()
            self.tbuf[self.guide_bit][index] = txt_cls_guide.detach()
            _B_batch[self.guide_bit] = self.bbuf[self.guide_bit][index]

            ALL_LOSS_DICT = self.compute_loss(output_dict, label, _B_batch, m1, m2)

            loss = 0
            for key in ALL_LOSS_DICT:
                loss += ALL_LOSS_DICT[key]

                if key in epoch_avg_loss_dict:
                    epoch_avg_loss_dict[key] += ALL_LOSS_DICT[key]
                else:
                    epoch_avg_loss_dict[key] = ALL_LOSS_DICT[key]
            epoch_avg_loss_dict['all_loss'] += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # undate B.
        for one in [self.bit, self.guide_bit]:
            self.bbuf[one] = torch.sign(self.ibuf[one] + self.tbuf[one])
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] all loss: {epoch_avg_loss_dict['all_loss'].data / (len(self.train_loader))}")
        self.logger.info(f"lr: {'-'.join([str('%.9f' % itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}")

    def train(self):
        self.logger.info("Start train...")

        total_train = 0.0
        total_valid = 0.0
        for epoch in range(self.args.epochs):
            epoch += 1
            time1 = time.time()
            self.train_epoch(epoch)

            time2 = time.time()
            train_time = round(time2 - time1, 2)

            if epoch % self.args.valid_freq == 0:
                self.valid(epoch)

            time3 = time.time()
            valid_time = round(time3 - time2, 2)

            total_train += train_time
            total_valid += valid_time

            self.logger.info(f"{self.args.dataset}. Train epoch [{epoch}], spend {train_time} sec")
            self.logger.info(f"{self.args.dataset}. Valid epoch [{epoch}], spend {valid_time} sec")

        self.logger.info(f">>>>>>> FINISHED {self.args.dataset}_full={self.args.full_ratio}_{self.bit}bit. <<<<<<<")
        self.logger.info(f"Best epoch: {self.best_epoch}, best avg_mAP: {self.max_avg}\n")
        self.logger.info(f"Train, avg spend {total_train / self.args.epochs} sec")
        self.logger.info(f"Valid, avg spend {total_valid / (self.args.epochs / 3.0)} sec")
        clear_logger()

    def valid(self, epoch):
        self.logger.info("\n")
        self.logger.info(" Valid: %d/%d " % (epoch, self.args.epochs))
        self.change_state(mode="valid")

        q_i, q_t = self.get_code(self.query_loader, self.args.query_num)
        r_i, r_t = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        _k_ = None
        m1_bool = self.d_m1.squeeze(-1).bool()
        m2_bool = self.d_m2.squeeze(-1).bool()
        r_i = r_i[m1_bool]
        r_t = r_t[m2_bool]
        r_i_label = self.retrieval_labels[m1_bool]
        r_t_label = self.retrieval_labels[m2_bool]

        mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                            r_t_label.to(self.device), _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                            r_i_label.to(self.device), _k_).item()
        avg_map = (mAPi2t + mAPt2i) / 2.0

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}]")
        self.logger.info(f"{self.bit} bits: MAP(i->t): {round(mAPi2t, 4)}, MAP(t->i): {round(mAPt2i, 4)}, Avg_map: {round(avg_map, 4)}")

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}]")
        self.logger.info(f"dataset={self.args.dataset}  full={self.args.full_ratio}  Avg mAP: {round(avg_map, 4)}")

        if avg_map > self.max_avg:
            self.best_epoch = epoch
            self.max_i2t = mAPi2t
            self.max_t2i = mAPt2i
            self.max_avg = avg_map
            # self.logger.info("$$$$$$$$$$$$$$$$$$$$ Best avg maps. $$$$$$$$$$$$$$$$$$$$$$$$")
            # self.save_model(epoch)
            # _file_name = "PCEH-" + self.args.dataset + "-" + "full:" + str(self.args.full_ratio) + "-" + str(self.bit) + ".mat"
            # self.save_mat(q_i, q_t, r_i, r_t, r_i_label, r_t_label)
            # self.logger.info(f"Save best *.mat data!")

        self.logger.info(f"Best epoch: {self.best_epoch}, best avg_mAP: {round(self.max_avg, 4)}")

        if not os.path.exists(os.path.join(self.args.save_dir, 'max_map.txt')):
            # 文件不存在，以写模式打开并写入时间戳
            with open(os.path.join(self.args.save_dir, 'max_map.txt'), 'a') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Created at: {timestamp}\n")

        with open(os.path.join(self.args.save_dir, 'max_map.txt'), 'a') as f:
            f.write('valid epoch: %d | map(i->t): %3.4f, map(t->i): %3.4f, avg_map: %3.4f\n' % (epoch, mAPi2t, mAPt2i, avg_map))

        if epoch == self.args.epochs:
            with open(os.path.join(self.args.save_dir, 'max_map.txt'), 'a') as f:
                f.write('====== dataset=%s  full=%.2f  best_epoch=%d ======\n' % (self.args.dataset, self.args.full_ratio, self.best_epoch))
                f.write('%3d bit: max_i2t: %3.4f, max_t2i: %3.4f, max_avg: %3.4f\n' % (self.bit, self.max_i2t, self.max_t2i, self.max_avg))
                f.write('==================================================\n\n')

    def predict_loss(self, pre_feat, ori_feat, mask):
        reconstruction_criterion = nn.MSELoss()
        pair_pre_feat = torch.mul(pre_feat, mask)
        pair_ori_feat = torch.mul(ori_feat, mask)
        loss = reconstruction_criterion(pair_pre_feat, pair_ori_feat)
        return loss

    def hash_loss_group(self, hi, ht, hi_buffer, ht_buffer, label_sim, B, K, weight=1, type='-16-bits'):
        ALL_LOSS = {}

        # CLS Intra -lambda1
        hyper_cls_intra = self.args.hyper_cls_intra
        ALL_LOSS[f'cls_intra_i_{type}'] = weight * hyper_cls_intra * self.bayesian_loss(hi_buffer, hi, label_sim)
        ALL_LOSS[f'cls_intra_t_{type}'] = weight * hyper_cls_intra * self.bayesian_loss(ht_buffer, ht, label_sim)

        # CLS Inter -lambda2
        hyper_cls_inter = self.args.hyper_cls_inter
        ALL_LOSS[f'cls_inter_likelihood_{type}'] = weight * hyper_cls_inter * \
                                                   (self.bayesian_loss(hi_buffer, ht, label_sim) + \
                                                    self.bayesian_loss(ht_buffer, hi, label_sim))

        # quantization loss
        ALL_LOSS[f'quantization_{type}'] = weight * (self.quantization_loss(hi, B, K_bits=K)
                                                     + self.quantization_loss(ht, B, K_bits=K))

        return ALL_LOSS

    def compute_loss(self, output_dict, label, B_batch, m1, m2):
        ALL_LOSS = {}

        label_sim = calc_neighbor(self.train_labels.float(), label)

        img_cls_hash = output_dict['img_cls_hash']
        txt_cls_hash = output_dict['txt_cls_hash']

        # main hash preser
        _loss_dict_group = self.hash_loss_group(img_cls_hash,
                                                txt_cls_hash,
                                                self.ibuf[self.bit],
                                                self.tbuf[self.bit],
                                                label_sim,
                                                B_batch[self.bit],
                                                K=self.bit,
                                                weight=1,
                                                type=f"-{self.bit}-bits"
                                                )
        ALL_LOSS.update(_loss_dict_group)

        # guide hash
        _loss_dict_group_guide = self.hash_loss_group(output_dict['img_cls_guide'],
                                                     output_dict['txt_cls_guide'],
                                                     self.ibuf[self.guide_bit],
                                                     self.tbuf[self.guide_bit],
                                                     label_sim,
                                                     B_batch[self.guide_bit],
                                                     K=self.guide_bit,
                                                     weight=self.args.mu,
                                                     type=f"-{self.guide_bit}-bits"
                                                     )
        ALL_LOSS.update(_loss_dict_group_guide)

        # Reconstruction target...
        _recon_i = _recon_t = B_batch[self.guide_bit]
        # Reconstruction loss...
        img_cls_hash_recon = output_dict['img_cls_hash_recon']
        txt_cls_hash_recon = output_dict['txt_cls_hash_recon']

        mu = self.args.mu
        hyper_recon = self.args.hyper_recon

        ALL_LOSS[f'recon_i'] = mu * hyper_recon * (
            F.mse_loss(_recon_i, img_cls_hash_recon, reduction='sum')) / (img_cls_hash_recon.shape[0])
        ALL_LOSS[f'recon_t'] = mu * hyper_recon * (
            F.mse_loss(_recon_t, txt_cls_hash_recon, reduction='sum')) / (txt_cls_hash_recon.shape[0])

        # mutual predict loss
        ALL_LOSS['pre_loss'] = 0
        pre_i_feat = output_dict['pre_i_feat']
        pre_t_feat = output_dict['pre_t_feat']
        ori_img_feat = output_dict['ori_img_feat']
        ori_txt_feat = output_dict['ori_txt_feat']
        ALL_LOSS['pre_loss'] = self.args.alpha * self.predict_loss(pre_i_feat, ori_img_feat, m1) \
                               + self.args.alpha * self.predict_loss(pre_t_feat, ori_txt_feat, m2)

        # Contrastive Alignment loss
        after_res_img_cls = output_dict['after_res_img_cls']
        after_res_txt_cls = output_dict['after_res_txt_cls']
        ALL_LOSS['after_res_infoNCE'] = self.args.beta  * self.info_nce_loss(
            after_res_img_cls, after_res_txt_cls, temperature=self.args.tao)

        return ALL_LOSS

    def get_code(self, data_loader, length: int):

        ibuf = torch.empty(length, self.bit, dtype=torch.float).to(self.device)
        tbuf = torch.empty(length, self.bit, dtype=torch.float).to(self.device)

        for image, text, key_padding_mask, label, m1, m2, index in tqdm(data_loader):
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            index = index.numpy()

            m1 = m1.to(self.rank, non_blocking=True)
            m2 = m2.to(self.rank, non_blocking=True)
            image = torch.mul(image, m1.unsqueeze(-1).unsqueeze(-1))
            text = torch.mul(text, m2)

            output_dict = self.model(image, text, key_padding_mask, m1, m2)

            img_cls_hash = output_dict['img_cls_hash'].detach()
            txt_cls_hash = output_dict['txt_cls_hash'].detach()

            ibuf[index, :] = torch.sign(img_cls_hash)
            tbuf[index, :] = torch.sign(txt_cls_hash)

        return ibuf, tbuf

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model.pth"))
        self.logger.info(f"Save model to {os.path.join(self.args.save_dir, 'model.pth')}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, r_i_label, r_t_label, fname='hashcode'):

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels_img = r_i_label.numpy()
        retrieval_labels_txt = r_t_label.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l_i': retrieval_labels_img,
            'r_l_t': retrieval_labels_txt
        }
        scio.savemat(os.path.join(self.args.save_dir, fname + '.mat'), result_dict)

    def info_nce_loss(self, out_1, out_2, temperature=0.07):
        # out_*: ND
        bz = out_1.size(0)
        targets = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature

        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, targets)
        loss1 = F.cross_entropy(scores1, targets)

        return 0.5 * (loss0 + loss1)

    def bayesian_loss(self, a: torch.Tensor, b: torch.Tensor, label_sim: torch.Tensor):
        # a: ND
        # b: MD
        # label_sim: NM
        s = 0.5 * torch.matmul(a, b.t()).clamp(min=-64, max=64)

        b_loss = -torch.mean(label_sim * s - torch.log(1 + torch.exp(s)))
        return b_loss

    def quantization_loss(self, hash_feature, B, K_bits):
        return F.mse_loss(hash_feature, B, reduction='sum') / (hash_feature.shape[0]) / K_bits

    def test(self):
        if self.args.pretrained == "" or self.args.pretrained == "MODEL_PATH":
            self.logger.error("test step must load a model! please set the --pretrained argument.")
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")

        self.change_state(mode="valid")

        q_i, q_t = self.get_code(self.query_loader, self.args.query_num)
        r_i, r_t = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        m1_bool = self.d_m1.squeeze(-1).bool()
        m2_bool = self.d_m2.squeeze(-1).bool()
        r_i = r_i[m1_bool]
        r_t = r_t[m2_bool]
        r_i_label = self.retrieval_labels[m1_bool]
        r_t_label = self.retrieval_labels[m2_bool]

        _k_ = None

        mAPi2t = calc_map_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                            r_t_label.to(self.device), _k_).item()
        mAPt2i = calc_map_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                            r_i_label.to(self.device), _k_).item()

        avg_map = (mAPi2t + mAPt2i) / 2.0

        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, map_avg:{avg_map}")
        with open(os.path.join(self.args.save_dir, 'test_map.txt'), 'a') as f:
            f.write('====== dataset=%s  full=%.2f  best_epoch=%d ======\n' % (
                self.args.dataset, self.args.full_ratio, self.best_epoch))
            f.write('%3d bit: max_i2t: %3.4f, max_t2i: %3.4f, max_avg: %3.4f\n' % (
                self.args.bit, mAPi2t, mAPt2i, avg_map))
            f.write('==================================================\n\n')

        query_img = q_i.cpu().detach().numpy()
        query_txt = q_t.cpu().detach().numpy()
        retrieval_img = r_i.cpu().detach().numpy()
        retrieval_txt = r_t.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels_img = r_i_label.numpy()
        retrieval_labels_txt = r_t_label.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l_i': retrieval_labels_img,
            'r_l_t': retrieval_labels_txt
        }
        scio.savemat(os.path.join(self.args.save_dir, "hashcode-" + str(self.args.bit) + ".mat"), result_dict)
        self.logger.info(">>>>>> save all data!")
