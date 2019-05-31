from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, get_object_or_404
from .models import Contrast, Exp, Coordinates, Coordinates_holder
from django.urls import reverse
from django.views import generic
from django.utils import timezone
# from django.forms.models import inlineformset_factory
from .forms import WordListForm,ROIFormSet
import sys
from .tasks import make_contrast
import random
import json
from string import replace
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from make_roi_figure import make_roi_figure
#import pickle

def IndexView(request):
    return render(request, 'simulate/index.html')
    # template_name = 'simulate/index.html'
    # context_object_name = 'latest_question_list'
    #
    # def get_queryset(self):
    #     """Return the last five published questions."""
    #     return 0#Question.objects.filter(pub_date__lte=timezone.now()).order_by('-pub_date')[:5]


def make_all_experiments(request):
    filename = 'simulate/jsons/experiments.json'
    experiments = json.loads(open(filename).read())
    for exp_name, exp_struct in experiments.items():
        if Exp.objects.filter(name=exp_name).exists():
            e = Exp.objects.get(name=exp_name)
        else:
            e = Exp.objects.create(name = exp_name)
        contrast_links = dict()
        for contrast_name, contrast in exp_struct['contrasts'].items():
            names1 = ''
            for condition in contrast['condition1']:
                names1 += exp_struct['stimuli'][condition]['value']
            names1 = replace(names1,';',',')
            names2 = ''
            for condition in contrast['condition2']:
                names2 += exp_struct['stimuli'][condition]['value']
            names2 = replace(names2,';',',')
            # print names2
            c = Contrast(list1_name=', '.join(contrast['condition1']),
                          list1_text=names1,
                          list2_name=', '.join(contrast['condition2']),
                          list2_text=names2,
                          experiment_id = e.id,
                          figures_list = json.dumps(contrast['figures']),
                          replicated_figure = 'replicated/group_{0}_{1}_flatmap.png'.format(exp_name,contrast_name))
            description = '[{0}] - [{1}]'.format(', '.join(contrast['condition1']), ', '.join(contrast['condition2']) )
            names = c.get_str()
            c.MNIstr , c.subjstr, c.pmaps = run_contrast(names['list1'],names['list2'])
            c.save()
            contrast_links[description] = str(c.id)
        e.contrasts_res = json.dumps(contrast_links)
        e.authors = ", ".join(exp_struct["authors"])
        e.DOI = exp_struct["DOI"]
        e.title = exp_struct["title"]
        e.save()
    return render(request, 'experiment_list.html')


def experiment_list(request):
    all_exp = Exp.objects.filter()
    txt = ''
    template = '<br> <h4> <li> <a  href={0}> {1} </a> </li> </h4> <br> '
    tmp_dict = {}
    exp_names = ['Binder2005','Bedny2013','Barrosloscertales2012','Davis2004','Kaplan2016']
    for exp in all_exp:
        tmp_dict[exp.name]=template.format('/{0}/experiment/'.format(exp.id),exp.title)
    for exp_name in exp_names:
        txt+=tmp_dict[exp_name]
    #     txt+=template.format('/simulate/{0}/experiment/'.format(exp.id),exp.title)
    # print txt
    return render(request, 'simulate/experiment_list.html', {'txt':txt})


def experiment_view(request, c_id):
    exp = Exp.objects.get(pk=c_id)
    template = '<br> <h4>  <li> <a  target="_parent" href={0}> {1} </a> </li> </h4> '
    txt = ''
    links = json.loads(exp.contrasts_res)
    for name, link in links.items():
        txt+= template.format('/{0}/contrast/'.format(link),name)
    return render(request, 'simulate/experiment.html', {'name':exp.name,'title':exp.title,
                                                        'DOI':exp.DOI, 'authors':exp.authors,
                                                        'txt':txt})


def run_contrast(names1,names2, **kwargs):
    info = {'DOI': '',
         'contrasts': {'contrast1': {'condition1': ['cond1'],
           'condition2': ['cond2'],
           'coordinates': [],
           'figures': []}},
         'coordinate_space': 'mni',
         'stimuli': {'cond1': {'type': 'word_list',
           'value': names1 },
          'cond2': {'type': 'word_list', 'value': names2}}}
    info.update(kwargs)
    viewer = make_contrast.delay(info)
    result = viewer.wait()#result = tasks.make_contrast.AsyncResult(id=viewer.id, app=tasks.make_contrast)
    #result = tasks.make_contrast(dict(info = info), type = '1')
    #if viewer.ready():
    #	viewer = viewer.get()
    viewerstr = result['group']
    subjstr = []
    for subj in range(3):
        subjstr.append(result['s_{}'.format(subj+1)])
    return viewerstr, json.dumps(subjstr), result['pmaps']


def get_word_list(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = WordListForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            c = form.save()
            # print c.id
            return HttpResponseRedirect(reverse('simulate:contrast', args= (c.id,)))

    # if a GET (or any other method) we'll create a blank form
    else:
        form = WordListForm()

    return render(request, 'simulate/contrast_filler.html', {'form': form})


def contrast_view(request, c_id):
    print "in contrast view"
    c =  Contrast.objects.get(pk=c_id)
    names = c.get_str()
    if len(c.get_MNI_names()['Cstr'])==0:
        c.MNIstr , c.subjstr, c.pmaps = run_contrast(names['list1'],names['list2'],do_perm = names['do_perm'])
        c.save()
    names['c_id'] = c_id
    names['exp_id'] = c.experiment_id

    return render(request, 'simulate/contrast.html',names)


def MNI_view(request, c_id):
    c =  Contrast.objects.get(pk=c_id)
    names = c.get_MNI_names()
    return render(request, 'simulate/MNI.html',names)


def subj_view(request, c_id, subj_num):
    c =  Contrast.objects.get(pk=c_id)
    names = c.get_subj_names(subj_num)
    return render(request, 'simulate/subj_{0}.html'.format(subj_num),names)


def published_contrast_view(request,c_id):
    c = Contrast.objects.get(pk=c_id)
    e = Exp.objects.get(pk=c.experiment_id)
    print "--------"
    print c.figures_list
    print "--------"
    print c.figures_list
    figures_list = json.loads(c.figures_list)
    figures_html = ''
    template = "<img src='/static/simulate/{0}' style=\"width: 100%\" />"
    if len(figures_list)>0:
        if len(figures_list[0])>0:
            # template = "<img src=\"{{%static 'simulate/{0}'}}\" alt='My image' />"
            for fig in figures_list:
                figures_html += template.format(fig)
    published_figures_html = template.format(c.replicated_figure)
    print "--------"
    print ""
    print "--------"
    return render(request, 'simulate/published_contrast_view.html', {'name':e.name,'title':e.title,
                                                        'DOI':e.DOI, 'authors':e.authors,
                                                        'figures_html':figures_html,
                                                         'published_figures_html':published_figures_html })


def existing_roi_analyses(request,c_id):
    Coordinates_holders =  Coordinates_holder.objects.filter(contrast__pk=c_id)
    txt = ''
    template = '<br> <h4> <li> <a  href={0}> {1} </a> </li> </h4> <br> '
    #tmp_dict = {}
    for ch in Coordinates_holders:
        txt+=template.format('/{0}/contrast/{1}/coordinates'.format(c_id,ch.id),ch.title)
    return render(request, 'simulate/existing_roi_analyses.html', {'txt':txt})

#
# def manage_books(request, author_id):
#     author = Author.objects.get(pk=author_id)
#     BookInlineFormSet = inlineformset_factory(Author, Book, fields=('title',))
#     if request.method == "POST":
#         formset = BookInlineFormSet(request.POST, request.FILES, instance=author)
#         if formset.is_valid():
#             formset.save()
#             # Do something. Should generally end with a redirect. For example:
#             return HttpResponseRedirect(author.get_absolute_url())
#     else:
#         formset = BookInlineFormSet(instance=author)
#     return render(request, 'manage_books.html', {'formset': formset})

def get_ROI_list(request, c_id):
    contrast = Contrast.objects.get(pk=c_id)
    coordinate_holder = Coordinates_holder(contrast = contrast)
    # coordinate_form = ROIFormSet(coordinate_holder, Coordinates)#, fields=('title',))
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = ROIFormSet(request.POST, instance=coordinate_holder)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            coordinate_holder.save()
            form.save()
            # print c.id
            return HttpResponseRedirect(reverse('simulate:coordinates', kwargs= dict(c_id = c_id,ch_id = coordinate_holder.id)))

    # if a GET (or any other method) we'll create a blank form
    else:
        form = ROIFormSet()

    return render(request, 'simulate/ROI_filler.html', {'form2': form, 'c_id': c_id})


def coordinates_view(request, c_id, ch_id):
    coordinate_holder_object = Coordinates_holder.objects.get(pk=ch_id)
    if coordinate_holder_object.roi_image_filename=='':
        contrast_obj =  Contrast.objects.get(pk=c_id)
        result_file = contrast_obj.pmaps
        random_roi_file = contrast_obj.random_roi_file
        coordinates = Coordinates.objects.filter(coordinates_holder__pk=ch_id).order_by('name')
        roi_list = [c for c in coordinates]
        results = make_contrast.delay(dict(result_file=result_file,
                                                                     roi_list = roi_list,radius = 10,
                                                                     random_roi_file=random_roi_file), type='2')
        results = results.wait()
        random_roi_file = results[0]
        roi_image_filename = results[1]
        allmasks = results[2]
        if contrast_obj.random_roi_file=='':
            contrast_obj.random_roi_file = random_roi_file
            contrast_obj.save()
        coordinate_holder_object.roi_image_filename = roi_image_filename
        coordinate_holder_object.allmasks = allmasks
        coordinate_holder_object.save()
    names = dict()
    names['image'] = '/static/'+coordinate_holder_object.roi_image_filename[18:]
    masks = json.loads(coordinate_holder_object.allmasks)
    txt = ''
    template = '<br> <h4> <li> <a  href={0}  target="_blank" > {1} </a> </li> </h4> <br> '
    #tmp_dict = {}
    for m in sorted(masks.keys()):
        txt+=template.format('/static/'+masks[m][18:],m)
    names['txt'] = txt
    names['c_id'] = c_id
    return render(request, 'simulate/coordinates.html',names)


#     return HttpResponseRedirect(reverse('roi_state') + '?job=' + job.id)
#
#
# def roi_state(request):
#     if 'job' in request.GET:
#         job_id = request.GET['job']
#     else:
#         return HttpResponse('No job id given.')
    #job = AsyncResult(job.id)
    #   data = job.result or job.state
    # json_data = json.dumps(data)
    # fig = make_roi_figure(roi_list, results_all, random_sample)
    # canvas=FigureCanvas(fig)
    # response=HttpResponse(content_type='image/png')
    # canvas.print_png(response)

    #return HttpResponse(json_data, mimetype='application/json')




#         HttpResponse(json.dumps(data), mimetype='application/json')
#
#
#
#     data = job.result or job.state
#     results_all, all_masks, random_sample
#     fig = make_roi_figure(roi_list, results_all, random_sample)
#     canvas=FigureCanvas(fig)
#     response = HttpResponse(content_type='image/png')
#     canvas.print_png(response)
#     return response
#
#
# self.update_state(state='PROGRESS',
#                 meta={'current': i, 'total': len(filenames)})
#
# def do_work():
#     """ Get some rest, asynchronously, and update the state all the time """
#     for i in range(100):
#         sleep(0.1)
#         current_task.update_state(state='PROGRESS',
#             meta={'current': i, 'total': 100})
#
#
# def poll_state(request):
#     """ A view to report the progress to the user """
#     if 'job' in request.GET:
#         job_id = request.GET['job']
#     else:
#         return HttpResponse('No job id given.')
#
#     job = AsyncResult(job_id)
#     data = job.result or job.state
#     return HttpResponse(json.dumps(data), mimetype='application/json')
#
#
# def init_work(request):
#     """ A view to start a background job and redirect to the status page """
#     job = do_work.delay()
#     return HttpResponseRedirect(reverse('poll_state') + '?job=' + job.id)
