import argparse
### Parsing arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and test CNN for ice-cube data", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    add_arg('--train','-tr',  action='store_false' ,dest='train_status' ,help='Has the model been trained?')
    add_arg('--test', '-ts',  action='store_false' ,dest='test_status'  ,help='Has the model been tested?')
    add_arg('--model_list', '-mdlst', nargs='+', type=int, dest='mod_lst',help=' Enter the list of model numbers to test ', required=True)
    return parser.parse_args()

if __name__=='__main__':
    args=parse_args()
    print(args)
    print(args.mod_lst)
    model_lst=args.mod_lst
    print(model_lst)
