## Replicating Binder et al. 2005: abstract vs. concrete words
## this code snippet assumes that a model has been fit in the EACL 2014 tutorial framework
## it needs the following variables defined:
## udwt = un-delayed semantic model weights for each voxel
## eng1000 = english1000 SemanticModel object

# First define stimuli. These are copied from paper
concrete_words = "apple, autumn, banana, bridge, cabin, canal, carrot, casket, cement, chin, circus, cloud, coffee, engine, flea, infant, island, leaf, lemon, lion, liquid, meadow, napkin, noose, orange, parade, pencil, pigeon, planet, pliers, plug, pole, potato, record, ring, saloon, sponge, spoon, steam, summer, sunset, swamp, tennis, toilet, tomato, truck, vodka, walnut, window, yacht".split(", ")
abstract_words = "advice, affair, aspect, assent, belief, blame, claim, custom, debut, deceit, desire, dogma, enigma, event, excuse, factor, gain, gist, glut, guess, heresy, hybrid, issue, item, malice, manner, method, motive, origin, output, outset, pardon, phase, plea, realm, regret, result, rumor, scheme, scorn, soul, supply, tale, tenure, theory, topic, treaty, upkeep, virtue, vista".split(", ")

# Find intersection of stimulus words with english1000 vocabulary
concrete_eng1000_words = [w for w in concrete_words if w in eng1000.vocab]
abstract_eng1000_words = [w for w in abstract_words if w in eng1000.vocab]

print len(concrete_eng1000_words), len(concrete_words)
print len(abstract_eng1000_words), len(abstract_words)

# Extract semantic vectors for each word
concrete_vecs = np.vstack([eng1000[w] for w in concrete_eng1000_words])
abstract_vecs = np.vstack([eng1000[w] for w in abstract_eng1000_words])

# Simulate responses to average word vectors
concrete_resp = np.dot(concrete_vecs.mean(0), udwt)
abstract_resp = np.dot(abstract_vecs.mean(0), udwt)

# Simulate contrast by subtracting responses
concrete_minus_abstract = concrete_resp - abstract_resp

# Visualize results
cortex.quickshow(cortex.Volume(concrete_minus_abstract, "S1", "20110321JG_auto2", mask=mask), 
                         with_rois=False, with_labels=False)

cortex.webshow(cortex.Volume(concrete_minus_abstract, "S1", "20110321JG_auto2", mask=mask))
