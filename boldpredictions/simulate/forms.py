from django.forms import ModelForm, Textarea
from .models import Contrast, Coordinates, Coordinates_holder
from django.forms.models import inlineformset_factory


# class WordListForm(forms.Form):
#     name1 = forms.CharField(label='Enter name of Condition 1', max_length=200)
#     word_list_1 = forms.CharField(label='Enter stimulus words seperated by a comma',
#                                   max_length=10000, widget=forms.Textarea)
#     name2 = forms.CharField(label='Enter name of Condition 2', max_length=200)
#     word_list_2 = forms.CharField(label='Enter stimulus words seperated by a comma',
#                                   max_length=10000, widget=forms.Textarea)

# from django import forms
# from xorformfields.forms import (
#     FileOrURLField, MutuallyExclusiveRadioWidget,
#     MutuallyExclusiveValueField, FileOrURLWidget,
#     )
# for _html_output
from django.utils.html import conditional_escape, html_safe
from django.utils.encoding import force_text, python_2_unicode_compatible
from django.utils import six
from django.utils.safestring import mark_safe


class WordListForm(ModelForm):

    class Meta:
        model = Contrast
        fields = ['list1_name', 'list1_text', 'baseline_choice', 'list2_name', 'list2_text', 'permutation_choice']
        # boolfield = forms.TypedChoiceField(
        #            coerce=lambda x: x == 'True',
        #            choices=((False, 'False'), (True, 'True')),
        #            widget=forms.RadioSelect
        #         )
        # test_field = MutuallyExclusiveValueField(
        #         fields=(forms.IntegerField(), forms.IntegerField()),
        #         widget=MutuallyExclusiveRadioWidget(widgets=[
        #             forms.Select(choices=[(1, 1), (2, 2)]),
        #             forms.TextInput(attrs={'placeholder': 'Enter a number'}),
        #         ]))
        widgets = {
            'list1_name': Textarea(attrs ={'cols':50, 'rows':1}),
            'list2_name': Textarea(attrs ={'cols':50, 'rows':1}),
            'list1_text': Textarea(attrs ={'cols':50, 'rows':10}),
            'list2_text': Textarea(attrs ={'cols':50, 'rows':10})
        }

    def _html_output2(self, normal_row, button_row, error_row, row_ender, help_text_html, errors_on_separate_row):
        "Helper function for outputting HTML. Used by as_table(), as_ul(), as_p()."
        top_errors = self.non_field_errors()  # Errors that should be displayed above all fields.
        output, hidden_fields = [], []

        for name, field in self.fields.items():
            html_class_attr = ''
            bf = self[name]
            # Escape and cache in local variable.
            bf_errors = self.error_class([conditional_escape(error) for error in bf.errors])
            if bf.is_hidden:
                if bf_errors:
                    top_errors.extend(
                        [_('(Hidden field %(name)s) %(error)s') % {'name': name, 'error': force_text(e)}
                         for e in bf_errors])
                hidden_fields.append(six.text_type(bf))
            else:
                # Create a 'class="..."' attribute if the row should have any
                # CSS classes applied.
                css_classes = bf.css_classes()
                if css_classes:
                    html_class_attr = ' class="%s"' % css_classes

                if errors_on_separate_row and bf_errors:
                    output.append(error_row % force_text(bf_errors))

                if bf.label:
                    label = conditional_escape(force_text(bf.label))
                    label = bf.label_tag(label) or ''
                else:
                    label = ''

                if field.help_text:
                    help_text = help_text_html % force_text(field.help_text)
                else:
                    help_text = ''

                if field.widget.__class__.__name__ == 'CheckboxInput':
                    output.append(button_row % {
                        'errors': force_text(bf_errors),
                        'label': force_text(label),
                        'field': six.text_type(bf),
                        'help_text': help_text,
                        'html_class_attr': html_class_attr,
                        'css_classes': css_classes,
                        'field_name': bf.html_name,
                    })
                else:
                    output.append(normal_row % {
                        'errors': force_text(bf_errors),
                        'label': force_text(label),
                        'field': six.text_type(bf),
                        'help_text': help_text,
                        'html_class_attr': html_class_attr,
                        'css_classes': css_classes,
                        'field_name': bf.html_name,
                    })

        if top_errors:
            output.insert(0, error_row % force_text(top_errors))

        if hidden_fields:  # Insert any hidden fields in the last row.
            str_hidden = ''.join(hidden_fields)
            if output:
                last_row = output[-1]
                # Chop off the trailing row_ender (e.g. '</td></tr>') and
                # insert the hidden fields.
                if not last_row.endswith(row_ender):
                    # This can happen in the as_p() case (and possibly others
                    # that users write): if there are only top errors, we may
                    # not be able to conscript the last row for our purposes,
                    # so insert a new, empty row.
                    last_row = (normal_row % {
                        'errors': '',
                        'label': '',
                        'field': '',
                        'help_text': '',
                        'html_class_attr': html_class_attr,
                        'css_classes': '',
                        'field_name': '',
                    })
                    output.append(last_row)
                output[-1] = last_row[:-len(row_ender)] + str_hidden + row_ender
            else:
                # If there aren't any rows in the output, just append the
                # hidden fields.
                output.append(str_hidden)
        return mark_safe('\n'.join(output))


    def as_p2(self):
        "Returns this form rendered as HTML <p>s."
        return self._html_output2(
            normal_row='<p%(html_class_attr)s>%(label)s <br> %(field)s%(help_text)s</p>',
            button_row='<p%(html_class_attr)s>%(field)s%(help_text)s %(label)s </p>',
            error_row='%s',
            row_ender='</p>',
            help_text_html=' <span class="helptext">%s</span>',
            errors_on_separate_row=True)

  #
  #
  #     <div class="form-group">
  #   <label for="inputEmail3" class="col-sm-2 control-label">Email</label>
  #   <div class="col-sm-10">
  #     <input type="email" class="form-control" id="inputEmail3" placeholder="Email">
  #   </div>
  # </div>

class ROI(ModelForm):

    class Meta:
        model = Coordinates
        fields = ['name', 'x', 'y', 'z']
        # boolfield = forms.TypedChoiceField(
        #            coerce=lambda x: x == 'True',
        #            choices=((False, 'False'), (True, 'True')),
        #            widget=forms.RadioSelect
        #         )
        # test_field = MutuallyExclusiveValueField(
        #         fields=(forms.IntegerField(), forms.IntegerField()),
        #         widget=MutuallyExclusiveRadioWidget(widgets=[
        #             forms.Select(choices=[(1, 1), (2, 2)]),
        #             forms.TextInput(attrs={'placeholder': 'Enter a number'}),
        #         ]))
        widgets = {
            'name': Textarea(attrs ={'cols':15, 'rows':1,'style':'resize:none;color:black'}),
            'x': Textarea(attrs ={'cols':3, 'rows':1,'style':'resize:none;color:black'}),
            'y': Textarea(attrs ={'cols':3, 'rows':1,'style':'resize:none;color:black'}),
            'z': Textarea(attrs ={'cols':3, 'rows':1,'style':'resize:none;color:black'})
        }

ROIFormSet = inlineformset_factory(Coordinates_holder, Coordinates,
                                            form=ROI, extra=10)