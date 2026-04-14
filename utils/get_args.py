import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bit", type=str, default=64, help="length of hash code.")
    parser.add_argument("--guide-bit-dim", type=int, default=128, help="length of guide hash codes.")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--res-mlp-layers", type=int, default=2, help="the number of ResMLP blocks.")

    parser.add_argument("--valid-freq", type=int, default=3, help="To valid every $valid-freq$ epochs.")
    parser.add_argument("--rank", type=int, default=1, help="GPU rank")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clip-lr", type=float, default=0.000001, help="learning rate for CLIP in CMCL.")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate for other modules in CMCL.")

    parser.add_argument("--is-train", default=True, action="store_true")
    parser.add_argument("--is-freeze-clip", action="store_true")

    parser.add_argument("--tao", type=float, default=0.07, help="")

    parser.add_argument("--transformer-layers", type=int, default=1)

    ### loss weight ###
    # alpha
    parser.add_argument("--alpha", type=float, default=5, help="weight of the predict")
    # beta
    parser.add_argument("--beta", type=float, default=5, help="weight of the contrastive alignment loss.")
    # lambda1
    parser.add_argument("--hyper-cls-intra", type=float, default=0.005, help="weight of the intra-modal similarity preservation loss")
    # lambda2
    parser.add_argument("--hyper-cls-inter", type=float, default=5, help="weight of the inter-modal similarity preservation loss.")
    # mu
    parser.add_argument("--mu", type=float, default=10, help="")
    # delta
    parser.add_argument("--hyper-recon", type=float, default=0.005, help="weight of the recon loss.")

    # other
    parser.add_argument("--clip-path", type=str, default="./ViT-B-32.pt", help="pretrained clip path.")
    parser.add_argument("--dataset", type=str, default="coco", help="choose from [coco, mirflickr, nuswide]")
    parser.add_argument("--query-num", type=int, default=5000)
    parser.add_argument("--train-num", type=int, default=10000)
    parser.add_argument('--full_ratio', type=float, default=0.3, help='complete pairs')
    parser.add_argument('--oimg_ratio', type=float, default=0.35, help='incomplete ratio with only images')

    parser.add_argument("--pretrained", type=str, default="", help="pretrained model path.")
    parser.add_argument("--index-file", type=str, default="mat/index.mat")
    parser.add_argument("--caption-file", type=str, default="mat/caption.mat")
    parser.add_argument("--label-file", type=str, default="mat/label.mat")
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-proportion", type=float, default=0.05,
                        help="Proportion of training to perform learning rate warmup.")

    args = parser.parse_args()

    return args
