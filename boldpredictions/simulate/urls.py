from django.conf.urls import url

from . import views

app_name = 'simulate'
urlpatterns = [
    url(r'^$', views.IndexView, name='index'),
    url(r'contrast_filler.html/$', views.get_word_list, name='contrast_filler'),
    url(r'experiment_list.html/$', views.experiment_list, name='experiment_list'),
    url(r'^(?P<c_id>[0-9]+)/experiment/$', views.experiment_view, name='experiment'),
    url(r'make_all_experiments.html/$', views.make_all_experiments, name='make_all_experiments'),
    url(r'^(?P<c_id>[a-zA-Z0-9]+)/contrast/$', views.contrast_view, name='contrast'),
    url(r'^(?P<c_id>[a-zA-Z0-9]+)/contrast/MNI.html$',views.MNI_view, name = 'MNI'),
    # url(r'^(?P<c_id>[a-zA-Z0-9]+)/contrast/first_four.html$',views.MNI_view, name = 'first_four'),
    url(r'^(?P<c_id>[a-zA-Z0-9]+)/contrast/subj(?P<subj_num>[0-9]+).html$',views.subj_view, name = 'subj'),
    url(r'^(?P<c_id>[a-zA-Z0-9]+)/contrast/published_contrast_view.html$',views.published_contrast_view,
        name ='published_contrast_view'),
    #url(r'roi_filler.html/$', views.get_ROI_list, name='roi_filler'),
    # url(r'^(?P<c_id>[a-zA-Z0-9]+)/contrast/(?P<form_id>[a-zA-Z0-9]+)/make_roi_plot.html$',views.make_roi_plot,
    #     name = 'make_roi_plot'),
    url(r'^(?P<c_id>[a-zA-Z0-9]+)/contrast/roi_filler.html$',views.get_ROI_list, name = 'roi_filler'),
    url(r'(?P<c_id>[a-zA-Z0-9]+)/contrast/existing_roi_analyses.html/$', views.existing_roi_analyses,
        name='existing_roi_analyses'),
    url(r'(?P<c_id>[a-zA-Z0-9]+)/contrast/(?P<ch_id>[a-zA-Z0-9]+)/coordinates.html/$', views.coordinates_view,
        name='coordinates'),
    # url(r'^(?P<c_id>[0-9]+)/MNIwebviewer/$', views.webviewer_view, name='MNIwebviewer')
]
