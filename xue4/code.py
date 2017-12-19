#_*_coding:utf-8_*_

f = open('allsens.txt', 'r')
lines = f.readlines()
f.close()
f2 = open('allsens_final.txt', 'w')

for line in lines:
    tmp = ''
    for i in range(len(line)):
        if ord(line[i]) >= 0 and ord(line[i]) <= 128:
            tmp += line[i]

    f2.write(tmp)
f2.close()

