
class Result(object):
    def __init__(self, html, data):
        self.html = html
        self.data = data

    @classmethod
    def combine(cls, results, wrap=None):
        """Combine a collection of Result objects into a single Result object.
        """
        combined_html = "\n".join([r.html for r in results])
        combined_data = [r.data for r in results]

        if wrap is not None:
            prefix, postfix = wrap
            combined_html = prefix + combined_html + postfix

        return cls(combined_html, combined_data)

    @classmethod
    def empty(cls):
        return cls('', None)
