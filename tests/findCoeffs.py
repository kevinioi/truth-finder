import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

# coefficient, line
reject = [100, 0]
support = [-100, 0]

with open("/Users//kevin//Documents//Ryerson//GitHub//Other//data/coefficients.csv", 'r') as doc:
    for i, line in enumerate(doc):
        try:
            value = float(line)

            if value > support[0]:
                support[0] = value
                support[1] = i + 1
            if value < reject[0]:
                reject[0] = value
                reject[1] = i + 1
        except:
            continue

print(reject)
print(support)