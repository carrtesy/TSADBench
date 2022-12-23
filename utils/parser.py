import argparse
import os
import torch

def prepare_arguments(parser):
    '''
    get input arguments
    :return: args
    '''
    # data
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
    parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")

    # save
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--logs", type=str, default="./logs")
    parser.add_argument("--outputs", type=str, default="./outputs")

    # thresholding
    parser.add_argument("--thresholding", type=str, default="oracle")

    # subparser of models
    subparser = parser.add_subparsers(dest='model')

    ## OCSVM
    OCSVM_parser = subparser.add_parser("OCSVM")
    OCSVM_parser.add_argument("--max_iter", type=int, default=300)
    OCSVM_parser.add_argument("--n_jobs", type=int, default=1)

    ## IsolationForest
    IsolationForest_parser = subparser.add_parser("IsolationForest")
    IsolationForest_parser.add_argument("--n_estimators", type=int, default=100,
                                        help=f"The number of base estimators in the ensemble.")
    IsolationForest_parser.add_argument("--n_jobs", type=int, default=1, help=f"The number of jobs to run")
    IsolationForest_parser.add_argument("--random_state", type=int, default=42, help=f"Random State")
    IsolationForest_parser.add_argument("--verbose", type=int, default=1, help=f"Verbose")

    ## LOF
    LOF_parser = subparser.add_parser("LOF")
    LOF_parser.add_argument("--n_jobs", type=int, default=1, help=f"The number of jobs to run")

    ## AE
    AE_parser = subparser.add_parser("AE")
    AE_parser.add_argument("--latent_dim", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")

    ## AECov
    AE_parser = subparser.add_parser("AECov")
    AE_parser.add_argument("--latent_dim", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")
    AE_parser.add_argument("--loss_fn", type=str, default="MSC-MCD", help=f"loss function")
    AE_parser.add_argument("--LAMBDA", type=int, default=0, help=f"Encoder, decoder hidden dim")
    AE_parser.add_argument("--OMEGA", type=int, default=0, help=f"Encoder, decoder hidden dim")

    ## VAE
    VAE_parser = subparser.add_parser("VAE")
    VAE_parser.add_argument("--latent_dim", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")
    VAE_parser.add_argument("--beta", type=float, required=True, default=1e-05, help=f"Encoder, decoder hidden dim")

    ## LSTMEncDec
    LSTMEncDec_parser = subparser.add_parser("LSTMEncDec")
    LSTMEncDec_parser.add_argument("--latent_dim", type=int, default=128, help=f"Encoder, decoder hidden dim")
    LSTMEncDec_parser.add_argument("--num_layers", type=int, default=3,
                                   help=f"The number of hidden layers.")
    LSTMEncDec_parser.add_argument("--dropout", type=float, default=0.1, help=f"dropout")
    LSTMEncDec_parser.add_argument("--subseq_length", type=float, default=100, help=f"subseq length when calculating mahalanobis distance (for cov matrix computation)")

    ## USAD
    USAD_parser = subparser.add_parser("USAD")
    USAD_parser.add_argument("--latent_dim", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")
    USAD_parser.add_argument("--alpha", type=float, required=False, default=0.1, help=f"alpha")
    USAD_parser.add_argument("--beta", type=float, required=False, default=0.9, help=f"beta")
    USAD_parser.add_argument("--dsr", type=int, default=5, help="down sampling rate")

    ## OmniAnomaly
    OmniAnomaly_parser = subparser.add_parser("OmniAnomaly")
    OmniAnomaly_parser.add_argument("--hidden_dim", type=int, default=128, help=f"Encoder, decoder hidden dim")
    OmniAnomaly_parser.add_argument("--z_dim", type=int, default=128, help=f"Encoder, decoder hidden dim")
    OmniAnomaly_parser.add_argument("--dense_dim", type=int, default=128, help=f"Encoder, decoder hidden dim")
    OmniAnomaly_parser.add_argument("--beta", type=int, default=1e-06, help=f"Initial Beta for KL Loss")

    ## AnomalyTransformer
    AnomalyTransformer_parser = subparser.add_parser("AnomalyTransformer")
    AnomalyTransformer_parser.add_argument('--anomaly_ratio', type=float, default=4.00)
    AnomalyTransformer_parser.add_argument('--k', type=int, default=3)
    AnomalyTransformer_parser.add_argument('--temperature', type=int, default=50)

    '''
    parse the arguments.
    '''
    args = parser.parse_args()
    args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
    args.logging_path = os.path.join(args.logs, f"{args.exp_id}")
    args.output_path = os.path.join(args.outputs, f"{args.exp_id}")

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.logging_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    args.home_dir = "."
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args




'''
Saved for future use.
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
'''
