from result import Result

class Repbase(object):

    @property
    def html_wrap(self):
        return ("<div class='{}'>".format(self.__class__.__name__), "</div>")

    def make_header(self):
        return Result.empty()

    def make_output(self, results):
        # Make header
        header = self.make_header()

        # Make sure this is a list
        if not isinstance(results, list):
            results = [results]

        # Combine results
        return Result.combine([header] + results, wrap=self.html_wrap)