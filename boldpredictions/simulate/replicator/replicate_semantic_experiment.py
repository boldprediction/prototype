from utils import read_json
from experiment import Experiment, ModelHolder
from subject import Subject
from subject_group import SubjectGroup
from result import Result
# import numpy as np


class Replicate(object):

    def __init__(self, experiments_json, subjects_json, model_dir, subject_analyses,
                 group_analyses, image_dir, jsons_dir="./jsons",
                 model_type="english1000"):

        self.model_type = model_type

        all_exp_info = read_json(experiments_json, jsons_dir)
        # exp2 = dict()
        # exp2['Binder2005'] = all_exp_info['Binder2005']
        # all_exp_info = exp2
        #all_exp_info2 = dict()
        #all_exp_info2['Bedny2013'] = all_exp_info['Bedny2013']
        #all_exp_info = all_exp_info2
        model_holder = ModelHolder(jsons_dir)
        self.experiments = {key: Experiment(model_holder = model_holder, name = key,  image_dir = image_dir,
                                            model_type=self.model_type, nperm = 1000,
                                            **info)
                            for (key, info) in all_exp_info.items()}

        subjects_info = read_json(subjects_json, jsons_dir)
        
        subjects = [Subject(name=key, model_type=self.model_type,
                            model_dir=model_dir['subject_dir'],
                            analyses=subject_analyses, **info)
                    for (key, info) in subjects_info.items()]

        self.subject_group = SubjectGroup(subjects, group_analyses)

    def run(self, sort_by='name'):
        """
        this is to run all results
        """
        if sort_by == 'name':
            key = lambda exp: exp.name

        sorted_experiments = sorted(self.experiments.values(), key=key)
        experiment_results = [exp.run(self.subject_group)
                              for exp in sorted_experiments]
        return Result.combine(experiment_results)


if __name__ == '__main__':

    import os
    import shutil
    from utils import load_config

    config = load_config()
    build_dir =  config["build_dir"]
    build_dir = build_dir+"replicator_website"
    model_dir = config["model_dir"]
    experiment_name = 'plan_experiment'#'experiments'#'plan_experiment'#'experiments'#'experiments_paper'#'experiments_paper'#'experiments'
    model_type = 'english1000'

    style = 'resources/css/style.css'

    static_dir = build_dir+"/static"
    main_file = os.path.join(build_dir, "index.html")

    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    if not os.path.exists(os.path.join(build_dir, static_dir)):
        os.mkdir(os.path.join(build_dir, static_dir))

    restrict_labels=['Broca', 'AC', 'FFA', 'EBA', 'OPA','PPA','IPS','RSC', 'TOS',
                                                   'V1','V3']


    from subject_analyses import SubjectCoordinatePlot,Flatmap, ThreeD, WebGLStatic, TotalEffectSize, \
        CoordinateAnalysis, CoordinateAnalysisRank, PermutationTestFlatmap, EmptyAnalysis
    subject_analyses = [#Flatmap(build_dir, static_dir, with_labels=True,
                        #        with_rois=True, restrict_labels = restrict_labels, labelsize = 50,linewidth = 6,
                        #        vmin = -3, vmax = 3)#, shadow = 1),
                        #ThreeD(build_dir, static_dir, pmap = True, vmin = -3, vmax = 3,  views = [0,5] ),
                        # PermutationTestFlatmap(build_dir, static_dir, with_labels=False,with_rois=True),
                        #PermutationTestFlatmap(build_dir, static_dir, with_labels=False,with_rois=True),
                        #SubjectCoordinatePlot(10, build_dir, static_dir, do6colors = False, with_labels=False,
                        #       with_rois=True, linecolor = (0,0,0,1))#, restrict_labels = restrict_labels, labelsize = 50,linewidth = 4)#****
                        EmptyAnalysis(),
                        #CoordinateAnalysis()
                        #CoordinateAnalysisRank()
                        #ThreeD(build_dir, static_dir),
                        # WebGLStatic(build_dir, static_dir)
                        ]

    from group_analyses import Mean, GroupCoordinateAnalysis, GroupCoordinateRank,GroupCoordinatePlot, TaskvsContrastPlot
    # group_analyses = [Mean(Flatmap(with_labels=False))]
    group_analyses = [
                         Mean([Flatmap(build_dir, static_dir, with_labels=False, with_rois=False),
                              #  ThreeD(build_dir, static_dir, vmin = -2, vmax = 2, views = [0,5]),
                              #ThreeD(build_dir, static_dir),
                             #GroupCoordinatePlot(15, build_dir, static_dir),***
                             # TaskvsContrastPlot(15, build_dir, static_dir),
                            #],smooth=None, pmap = True, do_1pct = False, mask_pred = False, recompute_mask = False),
                            #GroupCoordinatePlot(10, build_dir, static_dir),
                             # TaskvsContrastPlot(15, build_dir, static_dir),
                            ],smooth=None, pmap = True, do_1pct = False, mask_pred = False, recompute_mask = False),#** pmap = True
                         # Mean([Flatmap(build_dir, static_dir, with_labels=False, with_rois=False),
                         # #    #  ThreeD(build_dir, static_dir, vmin = -2, vmax = 2, views = [0,5]),
                         # #     #ThreeD(build_dir, static_dir),
                         # #     #GroupCoordinatePlot(15, build_dir, static_dir),***
                         # #     # TaskvsContrastPlot(15, build_dir, static_dir),
                         # #    #],smooth=None, pmap = True, do_1pct = False, mask_pred = False, recompute_mask = False),
                         # #     GroupCoordinatePlot(10, build_dir, static_dir),
                         # #   #  TaskvsContrastPlot(15, build_dir, static_dir),
                         #     ],smooth=None, pmap = False, do_1pct = False, mask_pred = False, recompute_mask = False),#** pmap = True
                         #]),
                      # Mean([#Flatmap(build_dir, static_dir, with_labels=False, with_rois=False),
                      # #      ThreeD(build_dir, static_dir),
                      #       GroupCoordinatePlot(15, build_dir, static_dir)
                      #       ],smooth=None, pmap = True, do_1pct = False, mask_pred = True),
                      # Mean([Flatmap(build_dir, static_dir, with_labels=False, with_rois=False),
                      #       GroupCoordinateAnalysis(15, build_dir, static_dir),
                      #      ],smooth=None, pmap = False),
                      # GroupCoordinateRank()
                      #ThreeD(build_dir, static_dir)
                      #WebGLStatic(build_dir, static_dir)
                      ]

    test_rep = Replicate('{0}.json'.format(experiment_name), 'subjects.json', model_dir,
                         subject_analyses, group_analyses, static_dir, jsons_dir = './jsons',model_type = model_type)
    result = test_rep.run()

    # print result.html

    # Write out main HTML
    with open(main_file, 'w') as out:
        out.write(result.html)

    # Copy style file
    shutil.copy(style, os.path.join(build_dir, os.path.basename(style)))
