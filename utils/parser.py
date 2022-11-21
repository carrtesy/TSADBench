import argparse
import os
import torch

def prepare_arguments(parser):
    ### data
    parser.add_argument("--dataset", type=str, required=True, default="SWaT", help=f"Dataset name")
    parser.add_argument("--batch_size", type=int, required=False, default=64, help=f"Batch size")
    parser.add_argument("--eval_batch_size", type=int, required=False, default=64 * 3, help=f"Batch size")
    parser.add_argument("--lr", type=float, required=False, default=1e-03, help=f"Learning rate")
    parser.add_argument("--window_size", type=int, required=False, default=12, help=f"window size")
    parser.add_argument("--stride", type=int, required=False, default=1, help=f"stride")
    parser.add_argument("--epochs", type=int, required=False, default=30, help=f"epochs to run")
    parser.add_argument("--load_pretrained", action="store_true", help=f"whether to load pretrained version")
    parser.add_argument("--exp_id", type=str, default="test")
    parser.add_argument("--scaler", type=str, default="std")
    parser.add_argument("--window_anomaly", action="store_true", help=f"window-base anomaly")
    parser.add_argument("--eval_every_epoch", action="store_true", help=f"evaluate every epoch")

    ### save
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--outputs", type=str, default="./outputs")

    ### thresholding
    parser.add_argument("--thresholding", type=str, default="oracle")

    ## subparser of models
    subparser = parser.add_subparsers(dest='model')
    OCSVM_parser = subparser.add_parser("OCSVM")
    IsolationForest_parser = subparser.add_parser("IsolationForest")
    LOF_parser = subparser.add_parser("LOF")
    USAD_parser = subparser.add_parser("USAD")

    ### USAD
    USAD_parser.add_argument("--latent_dim", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")
    USAD_parser.add_argument("--alpha", type=float, required=False, default=0.1, help=f"alpha")
    USAD_parser.add_argument("--beta", type=float, required=False, default=0.9, help=f"beta")
    USAD_parser.add_argument("--dsr", type=int, default=5, help="down sampling rate")
    USAD_parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")

    ### IsolationForest
    IsolationForest_parser.add_argument("--n_estimators", type=int, default=100,
                                        help=f"The number of base estimators in the ensemble.")
    IsolationForest_parser.add_argument("--n_jobs", type=int, default=-1, help=f"The number of jobs to run")
    IsolationForest_parser.add_argument("--random_state", type=int, default=42, help=f"Random State")
    IsolationForest_parser.add_argument("--verbose", type=int, default=1, help=f"Verbose")

    ### MAE
    MAE_parser = subparser.add_parser("MAE")
    MAE_parser.add_argument("--enc_dim", type=int, required=True, default=256, help=f"Encoder hidden dim")
    MAE_parser.add_argument("--dec_dim", type=int, required=True, default=256, help=f"Decoder hidden dim")
    MAE_parser.add_argument("--enc_num_layers", type=int, default=3, help=f"Encoder num layers")
    MAE_parser.add_argument("--dec_num_layers", type=int, default=3, help=f"Decoder num layers")
    MAE_parser.add_argument("--mask_ratio", type=float, default=0.1)
    MAE_parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")
    MAE_parser.add_argument("--eval_samples", type=int, default=3)

    ### VQVAE
    VQVAE_parser = subparser.add_parser("VQVAE")

    VQVAE_parser.add_argument("--n_embeddings", type=int, required=True, default=40,
                              help=f"Encoder, decoder hidden dim")
    VQVAE_parser.add_argument("--embedding_dim", type=int, required=True, default=40,
                              help=f"Encoder, decoder hidden dim")
    VQVAE_parser.add_argument("--beta", type=float, required=True, default=0.25, help=f"Encoder, decoder hidden dim")

    ### MADE
    MADE_parser = subparser.add_parser("MADE")
    MADE_parser.add_argument("--latent_dim", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")
    MADE_parser.add_argument("--depth", type=int, required=True, default=6, help=f"Encoder, decoder hidden dim")
    MADE_parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")

    ### ANP
    ANP_parser = subparser.add_parser("ANP")
    ANP_parser.add_argument("--dim_hid", type=int, required=True, default=256, help=f"Encoder hidden dim")
    ANP_parser.add_argument("--dim_lat", type=int, required=True, default=256, help=f"Decoder hidden dim")
    ANP_parser.add_argument("--mask_ratio", type=float, default=0.1)
    ANP_parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")
    ANP_parser.add_argument("--eval_samples", type=int, default=3)
    ANP_parser.add_argument("--masking_mode", type=str, default="continuous")

    args = parser.parse_args()
    args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
    args.output_path = os.path.join(args.outputs, f"{args.exp_id}")
    args.home_dir = "."
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args