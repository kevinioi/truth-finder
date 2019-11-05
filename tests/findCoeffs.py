import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')

# coefficient, line
reject = [100, 0]
support = [-100, 0]

with open("/Users//kevin//Documents//Ryerson//GitHub//Other//NLP_Project//coef", 'r') as doc:
    for i, line in enumerate(doc):

        value = float(line)

        if value > 0.3:
            print(value, i)

    
        if value < -0.4:
            print(value, i)

        #     if value < reject[0]:
        #         reject[1] = i
        #         reject[0] = value
        #     if value > support[0]:
        #         support[1] = i
        #         support[0] = value

print(reject)
print(support)