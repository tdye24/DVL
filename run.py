from utils.utils import *
from algorithm.vl.server import SERVER as VL_SERVER
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    args = parse_args()
    if args.use_wandb:
        import wandb

        wandb.init(project="VL", entity="tdye24")
        wandb.watch_called = False
        config = wandb.config
        args.exp_note = wandb.run.id
        config.update(args)
    else:
        config = args

    server = None
    if config.algorithm == 'vl':
        server = VL_SERVER(config=config)
    server.federate()
