from StringEncoder import StringEncoder
from StringEncoder import StringEncoder
labelEncoder = StringEncoder()

block_words = {'word1':1, 'word2':1}

def savelables(label, lines, urlsOnly=False):
	labelEncoded = labelEncoder.getCode(label)
	print label + ' : '+str(labelEncoded)

	for line in lines:
		words = line.replace('"','').replace('\r\n','').split(',')
		onlyWords = []
		for word in words:
			if not word.isdigit() and (len(word) > 2 and word[0] != '%' and word[1] != 'e' and word[2] != '0'):
				if word not in block_words:
					onlyWords.append(word)
	
		if len(onlyWords) > 0:
			fo.write(str(labelEncoded)+','+' '.join(onlyWords)+'\n')


def prepare_dataset():
	labels = ['Auto', 'Business', 'Entertainment', 'Health', 'Lifestyle', 'News', 'Sports', 'Technology', 'Travel']

	for label in labels:
		with open('/home/dev/ClassificationMulticlass/data/'+label+'.csv') as f:
			lines = f.readlines()
			savelables(label, lines, True)



if __name__ == "__main__":
	fo = open('dataset.txt', 'w')
	prepare_dataset()
	fo.close()
