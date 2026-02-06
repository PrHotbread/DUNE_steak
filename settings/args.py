import argparse

def drift():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-conv', 
        help="Stopping convergence criterion for the Finite Difference Method",
        default=1e-1, 
        required=False
    )
    parser.add_argument(
        '-namefile', 
        help="Name of the output binary file",
        default = "drift",
        required=False
    )
    parser.add_argument(
        '-s', 
        dest='setup', 
        help='Optional setup file', 
        default=''
    )
    return parser.parse_args()




def weighting():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-conv', 
        help="Stopping convergence criterion for the Finite Difference Method",
        default=1e-3, 
        required=False
    )

    parser.add_argument(
        '-view',
        help="Choses your view among: [view0, view1, view2]",
        choices=['view0', 'view1', 'view2'],
        required=True
    )

    parser.add_argument(
        '-namefile', 
        help="Name of the output binary file",
        default = "weighting",
        required=False
    )

    parser.add_argument(
        '-s', 
        dest='setup', 
        help='Optional setup file', 
        default=''
    )
    return parser.parse_args()