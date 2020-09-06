import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-F', '--file', type=str, default='')
opt = parser.parse_args()

input_filename = opt.file
output_filename = input_filename.split('.')[0]+'_display.md'
assert input_filename != ''
with open(input_filename, 'r') as f: 
    lines=f.readlines()      

res=[]
for idx, l in enumerate(lines): 
    flag=True 
    x='' 
    for i in range(len(l)): 
        if l[i]!='$' and (flag==True or (flag==False and l[i]!=' ' and l[i]!='(' and l[i]!=')')): 
            x+=l[i] 
        elif l[i]==' ': 
            x+='%20' 
        elif l[i]=='(': 
            x+='%28' 
        elif l[i]==')': 
            x+='%29' 
        else: 
            if flag==True: 
                x+='![latex_equ](https://latex.codecogs.com/svg.latex?' 
            else: 
                x+=')'                 
            flag=not flag 
            
    assert flag==True 
    res.append(x) 

with open(output_filename, 'w+') as f: 
    for l in res: 
        f.write(l)

print(len(lines), len(res), 'OK')
