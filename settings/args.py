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



def gen_signal():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', 
        help="Electrons number generated", 
        default = 100
    )

    parser.add_argument(
        '-output', 
        help = "Save the signal", 
        choices = [True, False], 
        default = False
    )

    parser.add_argument('-namefile', 
        help = "Give a name for the output files", 
        default = "out"
    )

    parser.add_argument('-traj', 
        help= "Displays electron trajectories. Warning: for electron numbers > 1000, this take some time.", 
        default = False
    )

    parser.add_argument('-csv', 
        help= "Save the start/end point and the charge induced for each electron. Warning: Size file > 1 Mo", 
        default = False
    )

    parser.add_argument('-s', 
        dest='setup', 
        help='Setup file, if needed', 
        default=''
    )
    return parser.parse_args()