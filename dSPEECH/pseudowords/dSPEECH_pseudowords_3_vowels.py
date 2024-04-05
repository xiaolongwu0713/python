from pprint import pprint
import random
import numpy as np
random.seed(42)

C = ['m', 'n', 'ng', 'p', 't', 'k']
V = ['ee', 'oo', 'a']

# No vowels in final (removes 'poo', 'pee', 'moo').
# Removed 'long' pseudowords (CVCVCVC).
# structure = { 1: ' CVC CVCVC VC ',
# 			  2: ' VC CVCVC CVC ',
# 			  3: ' CVCVC VC CVC ',
# 			  4: ' VC CVC CVCVC ',
# 			  5: ' CVC VC CVCVC ',
# 			  6: ' CVCVC CVC VC ',
# 			 }

structure = { 1: ' CVC VC ',
			  2: ' CVC VC ',
			  3: ' CVC VC ',
			  4: ' CVC VC ',
			  5: ' CVC VC ',
			  6: ' CVC VC ',
			 }
chars = [' ', 'm', 'n', 'ng', 'p', 't', 'k', 'ee', 'oo', 'a']
counts={i : 0 for i in chars}
imf =  {i : {'initial':0, 'medial':0, 'final':0} for i in chars}

iterations = 17 # Produces 102 pseudo-sentences (17 lots of 6 structures, above). Change this to produce more/less sentences.

sentence_list = []
for i in range(iterations):
	structures = random.sample([i for i in structure.keys()], len(structure))
	for j in structures:
		sentence = structure[j]
		flag = True
		while flag:
			Cs = random.sample(C, len(C)) # The consonants, at random
			Vs = random.sample(V, len(V)) + [V[int(np.floor((j-1)/2))]]# The vowels, at random, plus an iterative additional
			string = []
			for CV in range(len(sentence)):
				if sentence[CV]=='C':
					# handler for the 'ng' rule exception (only at word end, and only after /a/):
					if Cs[-1]=='ng':
						if string[CV-1]=='a' and sentence[CV+1]==' ':
							string.append(Cs[-1])
							del Cs[-1]
						else:
							string.append(None)
							del Cs[-1]
					else:
						string.append(Cs[-1])
						del Cs[-1]
				elif sentence[CV]=='V':
					string.append(Vs[-1])
					del Vs[-1]					
				else:
					string.append(' ')
			if None in string:
				pass
			else:
				string+=[' '] # padding
				joined = ''.join(string)
				words = joined.split()
				if 'took' in words: # took pronounced /tʊk/
					pass
				elif 'nook' in words: # took pronounced /nʊk/
					pass
				else:
					# initial,  medial, final position check:
					for letter in range(len(string)):
						if string[letter] in imf.keys():
							if string[letter-1]!=' ' and string[letter+1]!=' ':
								imf[string[letter]]['medial']+=1
							elif string[letter-1]==' ':
								imf[string[letter]]['initial']+=1
							elif string[letter+1]==' ':
								imf[string[letter]]['final']+=1
							else:
								pass
					for letter in string:
						counts[letter]+=1
					sentence_list.append(joined.strip())
					flag=False

del(counts[' '])
print('\n:Generated pseudo-sentences:')
pprint(sentence_list)
print('\nCount of phones in the generated pseudo-sentences:')
print(counts)
print('\nPositions of phones in the generated pseudo-sentences:')
pprint(imf)