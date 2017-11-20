## Why we need to extract texts from a given Video?
Getting the text contents like subtitles and background banner will give lot of context information. This will add lot of value to the video analysis system.

## Technology used:
We will use **tesseract**, a commercial quality OCR engine originally developed at HP between 1985 and 1995. In 1995, this engine was among the top 3 evaluated by UNLV. It was open-sourced by HP and UNLV in 2005, and has been developed at Google since then.
We use a python wrapped API of tesseract to recognize texts in given image.
Underlying OCR engine is based on LSTM neural networks. Model data for 101 languages is available in the tessdata repository.
Latest tersseract 4.0 is implemented in VGSL. VGSL (Variable-size Graph Specification Language) enables the specification of a neural network, composed of convolutions and LSTMs, that can process variable-sized images, from a very short definition string.

### Code:

```
from pytesseract import image_to_string 
import sys
from PIL import Image

imageFile = 'test.jpg'
print image_to_string(Image.open(imageFile),lang='eng')
```

## Spelling Corrector
Its really important to do a spelling correction on all the words being identified to improve correctness.
We used a probabilistic approach of finding correct spelling as described in http://norvig.com/spell-correct.html


### Code:

```
import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


print correction('speling')    #this will print 'spelling'

print correction('korrectud')   #this will print 'corrected'

```

## Reference:
- https://github.com/tesseract-ocr/tesseract
- https://github.com/tesseract-ocr/tesseract/blob/master/doc/tesseract.1.asc#languages
- http://norvig.com/spell-correct.html
