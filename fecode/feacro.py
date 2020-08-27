"""Find candidate acronyms"""
import argparse 
import sys
from operator import itemgetter

# expected to read acronym\tcontext lines, such as: 
# ICS      . ( INTEGRATED COOLING SYSTEM
# ICS      .. ( INTEGRATED COOLING SYSTEM
# ICS      2017 ( INTEGRATED COOLING SYSTEM

# ---------------------------------------------
#       ligne de commande
# ---------------------------------------------
# 
def get_args(): 
    parser = argparse.ArgumentParser(description='search for acronyms.')
    
    parser.add_argument("-v", '--verbosity', type=int, help="increase output verbosity", default=0)    
    parser.add_argument("-t", '--top', type=int, help="print top candidates per acronym", default=1) 
    parser.add_argument("-x", '--test', action='store_true' , help="only run tests", default=False) 
    parser.add_argument("-m", '--min', type=int, help="min score of a resolution", default=0) 

    args = parser.parse_args()
    return args

# ---------------------------------------------
#        main
# ---------------------------------------------
def main():
 

    args = get_args()
    nblines = 0
    nbsolvings = 0
    nbsolved = 0

    if args.test:
        print(solve('SATCOM',  'SATCOM INOP'))
        print(solve("WATER","WATER")) 
        print(solve("SVDU","BULKHEAD-MOUNTED SMART VIDEO DISPLAY UNIT")) 
        print(solve("EMK", "BLABLA EQUIPMENT - EMERGENCY MEDICAL KIT NARATIVE"))
        print(solve("EMK", "SMOK"))
    else:

        for line in sys.stdin:

            nblines += 1
            t = line.rstrip().split('\t')

            if len(t) == 2:

                acro = t[0].strip()
                context = t[1].strip()

                # remove badly scored ones, and remove the score of survivals
                sol = list(map(itemgetter(0),
                               filter(lambda x: x[1] >= args.min, 
                                      solve(acro,context, args.verbosity > 3))))

                nbsolvings += 1
                if len(sol) >0:
                    nbsolved += 1

                if args.verbosity:
                    print(f"#acro: {acro} context: {context}",file=sys.stderr)

                # print top ones    
                for a in sol[:args.top]:
                    print(f"{acro}\t{a}")


        print(f"#lines {nblines} #solvings {nbsolvings} #solved: {nbsolved}",file=sys.stderr)                

# ---------------------------------------------
#       could do much lighter 
# ---------------------------------------------
def solve(acro,context,verbose):

   l_acro = len(acro)
   indexes = []
   acros = []

   # currently, the score favours resolutions with letters at the beginning of words
   # and shorter resolutions
   # should be improved
   def _score():
      sc = 0
      for i in indexes:
        if i == 0 or context[i-1].isspace():
            sc += 1
      return sc * 1000 - len(_plain())
            
   # the resolution  (string of plain words)          
   def _plain():
    
      d = indexes[0]
      f = context.find(' ',indexes[-1])
      if verbose:
        print(f"plain: i:{indexes[-1]} d:{d} f:{f}")

      if d>0 and not context[d-1].isspace():
        d = context.rfind(' ',0,d-1)
        d = 0 if d == -1 else d+1
      #if verbose:
        #print(d,f)  
      return context[d:] if f == -1 else context[d:f]

   # the beats: nothing beats recursvity
   def _solve(i_acro, i_context):
     if i_acro == l_acro:         
        resolution = _plain()
        score = _score()
        acros.append((resolution,score))
        if verbose:     
            print(f"info [{acro}]\t[{context}]\t{indexes}\t{score}\t{resolution}", file=sys.stderr)
            for i in indexes:
                print('sfx: ',context[i:], file=sys.stderr)       
     else:
        a = acro[i_acro]
        i = context.find(a,i_context)
        while i != -1:
            indexes.append(i)
            _solve(i_acro+1,i+1)
            indexes.pop()
            i = context.find(a,i+1)

 
   _solve(0,0)

   # sort by score, strip score, and remove equalities
   return sorted(acros,key=itemgetter(1), reverse=True)

# ---------------------------------------------
#        
# ---------------------------------------------
if __name__ == '__main__':
    main()
