from repbase import Repbase
from result import Result

class SubjectGroup(list, Repbase):
    def __init__(self, subjects, analyses, do_pmap = False):
        self += subjects
        self.analyses = analyses

    def aggregate(self, subjectwise_results, contrast, do_pmap):
	if isinstance(self.analyses[0], (list)):
	    if do_pmap:
		return self.make_output([analysis(subjectwise_results, self,contrast)
                                 for analysis in self.analyses[1]])
	    else:
		return self.make_output([analysis(subjectwise_results, self,contrast)
                                 for analysis in self.analyses[0]])
	else:
            return self.make_output([analysis(subjectwise_results, self,contrast)
                                 for analysis in self.analyses])
    
    def run(self, contrast, do_pmap = False):
        """
        combine contrasts from subjects somehow
        :param contrast: Contrast object
        :return: visualization object with everything
        """
        # Make subject-wise results
        subject_output, subject_results = zip(*[sub.run(contrast, do_pmap = do_pmap ) for sub in self])
        subject_output = list(subject_output)

        # Make group results
        group_results = self.aggregate(subject_results,contrast, do_pmap = do_pmap)

        # Combine and return
        return self.make_output([group_results] + subject_output)

