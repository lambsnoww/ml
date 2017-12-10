f = open('emoji.txt', 'r')
a = f.readlines()
print len(a)
print a
f.close()
e = [int(i) for i in a]
print len(e)
print e