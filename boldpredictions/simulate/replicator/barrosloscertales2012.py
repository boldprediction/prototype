## Replicating Barros-Loscertales et al. 2012: gustatory vs. control words
## this code snippet assumes that a model has been fit in the EACL 2014 tutorial framework
## it needs the following variables defined:
## udwt = un-delayed semantic model weights for each voxel
## eng1000 = english1000 SemanticModel object

gustatory_words = "olive, anise, celery, anchovy, coffee, candy, onion, beer, chocolate, sausage, cauliflower, pork, fanta, flan, strawberry, cookie, chili, stew, ham, lemon, mint, hake, jam, swordfish, honey, mustard, cream, paella, bread, pasta, turkey, fish, pineapple, pizza, banana, chicken, cheese, salt, frankfurter, salami, sauce, sardine, cider, omelet, grape, wine, yogurt".split(", ")
control_words = "hoop, bus, aircraft, bamboo, tray, swimsuit, boat, trunk, blouse, button, brazier, buddha, armchair, cavalcade, cactus, coach, helmet, hut, kite, dart, axis, shield, bassoon, lighthouse, paper, bellows, crochet, jade, canvas, hank, cloak, motel, shovel, passport, hair, shutter, pole, skin, peg, pool, desk, radiator, rake, grill, sack, footpath, sofa, ring, rudder, railroad".split(", ")

gustatory_eng1000_words = [w for w in gustatory_words if w in eng1000.vocab]
control_eng1000_words = [w for w in control_words if w in eng1000.vocab]

print len(gustatory_eng1000_words), len(gustatory_words)
print len(control_eng1000_words), len(control_words)

gustatory_vecs = np.vstack([eng1000[w] for w in gustatory_eng1000_words])
control_vecs = np.vstack([eng1000[w] for w in control_eng1000_words])

gustatory_resp = np.dot(gustatory_vecs.mean(0), udwt)
control_resp = np.dot(control_vecs.mean(0), udwt)
gustatory_minus_control = gustatory_resp - control_resp

cortex.quickshow(cortex.Volume(gustatory_minus_control, "S1", "20110321JG_auto2", mask=mask), 
                 with_rois=False, with_labels=False);

cortex.webshow(cortex.Volume(gustatory_minus_control, "S1", "20110321JG_auto2", mask=mask))
