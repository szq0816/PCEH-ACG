from utils.get_args import get_args
from train_asym import TrainerAsym


def main(arg):
    trainer = TrainerAsym(arg)
    trainer.run()


if __name__ == "__main__":

    for dataset in ['mirflickr', 'nuswide', 'coco']:
        args = get_args()
        args.dataset = dataset
        if args.dataset == 'mirflickr':
            args.lr = 0.001
            args.mu = 0.5
            args.valid_freq = 3
            args.epochs = 30
            args.hyper_recon = 0.001
            args.query_num = 2000
            args.train_num = 10000
            args.caption_file = "mat/caption.mat"
        elif args.dataset == 'nuswide':
            args.lr = 0.002
            args.mu = 0.7
            args.valid_freq = 3
            args.epochs = 30
            args.hyper_recon = 0.001
            args.query_num = 2100
            args.train_num = 10500
            args.caption_file = "mat/caption.txt"
        elif args.dataset == 'coco':
            args.lr = 0.002
            args.mu = 10
            args.valid_freq = 3
            args.epochs = 30
            args.hyper_recon = 0.005
            args.query_num = 5000
            args.train_num = 10000
            args.caption_file = "mat/caption.mat"

        full = [0.1, 0.3, 0.5, 1.0]
        oimg = [0.45, 0.35, 0.25, 0.0]

        import datetime

        _time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if not args.is_train:
            _time += "_test"

        for i in [0, 1, 2]:
            args.full_ratio = full[i]
            args.oimg_ratio = oimg[i]

            for args.bit in [16, 32, 64]:
                args.save_dir = f"./result/{args.dataset}/PCEH-ACG/full:{args.full_ratio}/bit:{args.bit}"

                main(args)
