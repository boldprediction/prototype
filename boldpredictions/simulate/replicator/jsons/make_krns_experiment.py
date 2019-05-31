# script to make things easier

for i in range (1,49):
    print "      \"contrast_{0}\": {{".format(i)
    print "        \"condition1\" : [\"topic_{0}_1\",\"topic_{0}_2\"],".format(i)
    print "        \"condition2\" : [\"baseline\"],"
    print "        \"coordinates\": [ {\"name\": \"\","
    print "                         \"xyz\": [],"
    print "                         \"zscore\": []}],"
    print "        \"figures\": [\"\"]"
    print "      },"