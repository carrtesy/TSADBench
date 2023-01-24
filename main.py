######################################################
#                       _oo0oo_                      #
#                      o8888888o                     #
#                      88" . "88                     #
#                      (| -_- |)                     #
#                      0\  =  /0                     #
#                    ___/`---'\___                   #
#                  .' \\|     |// '.                 #
#                 / \\|||  :  |||// \                #
#                / _||||| -:- |||||- \               #
#               |   | \\\  -  /// |   |              #
#               | \_|  ''\---/''  |_/ |              #
#               \  .-\__  '-'  ___/-. /              #
#             ___'. .'  /--.--\  `. .'___            #
#          ."" '<  `.___\_<|>_/___.' >' "".          #
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |        #
#         \  \ `_.   \_ __\ /__ _/   .-` /  /        #
#     =====`-.____`.___ \_____/___.-`___.-'=====     #
#                       `=---='                      #
#                                                    #
#                                                    #
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    #
#                                                    #
#        Buddha Bless:   "No Bugs in my code"        #
#                                                    #
######################################################

import wandb
import hydra
from omegaconf import DictConfig

from utils.logger import make_logger
from utils.argpass import prepare_arguments
from utils.tools import SEED_everything

import warnings
from data.load_data import DataFactory
from Exp.SklearnBaselines import *
from Exp.ReconBaselines import *
from Exp.Baselines import *

SEED_everything(42)
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="cfgs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    # prepare arguments
    args = prepare_arguments(cfg)

    # WANDB
    wandb.login()
    WANDB_PROJECT_NAME, WANDB_ENTITY = "TSADBench", "carrtesy"
    wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name=args.exp_id)
    wandb.config.update(args)

    # Logger
    logger = make_logger(os.path.join(args.log_path, f'{args.exp_id}.log'))
    logger.info(f"Configurations: {args}")

    # Data
    logger.info(f"Preparing {args.dataset} dataset...")
    datafactory = DataFactory(args, logger)
    train_dataset, train_loader, test_dataset, test_loader = datafactory()
    args.num_channels = train_dataset.x.shape[1]

    # Model
    logger.info(f"Preparing {args.model.name} Trainer...")
    Trainers = {
        "RandomModel": RandomModel_Trainer,
        "OCSVM": OCSVM_Trainer,
        "IsolationForest": IsolationForest_Trainer,
        "LOF": LOF_Trainer,
        "LSTMEncDec": LSTMEncDec_Trainer,
        "LSTMVAE": LSTMVAE_Trainer,
        "USAD": USAD_Trainer,
        "OmniAnomaly": OmniAnomaly_Trainer,
        "DeepSVDD": DeepSVDD_Trainer,
        "DAGMM": DAGMM_Trainer,
        "THOC": THOC_Trainer,
        "AnomalyTransformer": AnomalyTransformer_Trainer,
    }

    trainer = Trainers[args.model.name](
        args=args,
        logger=logger,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    # 4. train
    logger.info(f"Preparing {args.model.name} Training...")
    trainer.train()

    # 5. infer
    logger.info(f"Loading from best path...")
    trainer.load(os.path.join(args.checkpoint_path, f"best.pth"))
    result = trainer.infer()

    # 6. final results
    wandb.log(result)
    logger.info(f"=== Final Result ===")
    for key in result:
        logger.info(f"{key}: {result[key]}")
        with open(os.path.join(args.result_path, f"{key}.txt"), 'w') as f:
            f.write(str(result[key]))

if __name__ == "__main__":
    main()