"""KeyWord In Context for (ACRO) strings"""
import argparse
import re
import sys
    
 
# ---------------------------------------------
#       ligne de commande
# ---------------------------------------------
# 
def get_args(): 
    parser = argparse.ArgumentParser(description='search for (acro) in stdin lines.')
    
    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0) 
    parser.add_argument("-s", '--size', type=int, help="# words to print before (ACRO)", default=5) 
    parser.add_argument("-m", '--min', type=int, help="min length of an acronym (not counting parenthes)", default=2) 
    parser.add_argument("-M", '--max', type=int, help="max length of an acronym (not counting parenthes)", default=5) 
    parser.add_argument("-a", '--alpha', action='store_true', help="enforce alpha only acronyms?", default=False) 

    args = parser.parse_args()
    return args

# ---------------------------------------------
#        main
# ---------------------------------------------
def main():
  
    args = get_args()
     
    # note: could relax this for more acronyms  
    p = re.compile('^\(\w+\)$')

    # iterate through stdin 
    for line in sys.stdin:    
 
        s = line.rstrip().split()
        l = len(s)

        if args.verbosity>2:
            print(s,file=sys.stderr)

        for i in range(l):
            acro = s[i][1:-1]
            if (not args.alpha) or acro.isalpha():
                if len(acro) >= args.min and len(acro) < args.max and not p.match(s[i]) is None:
                    if args.verbosity: 
                        print('in:',line.rstrip(),file=sys.stderr)
                    context = " ".join(s[max(0,i-args.size):i]).strip()               
                    if len(acro) < len(context):
                        print(acro,'\t',context) 

# ---------------------------------------------
#        
# ---------------------------------------------
if __name__ == '__main__':
    main()
