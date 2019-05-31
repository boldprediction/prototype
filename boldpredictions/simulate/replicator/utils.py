import json
import numpy as np
import os
import sys

from SemanticModel import SemanticModel

FSLDIR = os.getenv("FSLDIR")
if FSLDIR is None:
    import warnings
    warnings.warn("Can't find FSLDIR environment variable, assuming default FSL location..")
    #FSLDIR = "/usr/local/fsl-5.0.10"
    FSLDIR = "/usr/local/fsl"
    os.environ["FSLDIR"] = FSLDIR
    PATH = os.getenv("PATH")
    os.environ["PATH"] = PATH+":"+FSLDIR+"/bin"


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_json(filename, filepath="jsons"):
    """Returns a json files content."""
    filename = os.path.join(BASE_DIR,filepath, filename)
    return json.loads(open(filename).read())


def load_config(json_dir = 'jsons'):

    # FIXME: I actually would like to have the caller method name
    # and not the caller module name. But didn't find it yet.
    caller_object = sys._getframe(1).f_code
    caller_name = caller_object.co_filename
    print "load_config is called by {}".format(caller_name)

    """Checks if the config_user.json file exists and loads this. Otherwise
    loads the config_default.json file."""
    if os.path.exists(os.path.join(BASE_DIR,json_dir,"config_user.json")):
        print "User config file for directories is used."
        config = read_json("config_user.json", filepath = json_dir)
    else:
        print "Default config file for directories is used. If you want\
        to use your own directory structure create a config_user.json file\
        under ./jsons."
        config = read_json("config_default.json")
    return config


def load_model(model_type, json_dir = "jsons"):

    config = load_config(json_dir)
    model_dir = config["model_dir"][model_type]

    if model_type == 'english1000':
        print('\n Loaded english1000! \n')
        return SemanticModel.load(os.path.join(model_dir, "english1000sm.hf5"))
    elif model_type == 'word2vec':
        modelfile=os.path.join(model_dir, "GoogleNews-vectors-negative300.bin")
        norm=False
        from gensim.models.word2vec import Word2Vec
        model = Word2Vec.load_word2vec_format(modelfile, binary=True, max_vocab_size = 10000)
        usevocab = set(cPickle.load(open("/auto/k8/huth/storydata/comodels/complete2-15w-denseco-mat-vocab")))
        vocab, vocinds = zip(*[(w, model.vocab[w].index) for w in model.vocab])
        #w2v_usevocab = [(w,val.index) for w,val in w2v.vocab.items() if w in usevocab]
        #srtvocab = [w for w,voc in sorted(w2v.vocab.items(), key=lambda item:item[1].index)]
        #srtvocab,srtinds = zip(*sorted(w2v_usevocab, key=lambda item:item[1]))
        if norm:
            data = model.syn0norm[list(vocinds)]
        else:
            data = model.syn0[list(vocinds)]
        return  SemanticModel(data.T, vocab)
    else:
        raise ValueError('Unknown model type: %s' % self.model_type)


def mni2vox(mni_coord, transform):

    """Given an MNI coordinate (mm-space) and the affine transformation of the 
    image (nifti_image.get_affine()) this function returns the coordinates in 
    voxel space.

    """ 
    return np.array(mni_coord+[1]).dot(np.linalg.inv(transform).T)[:3]


def mni2vox_dumb(mni_coord, template="MNI152_T1_1mm_brain"):

    if template == "MNI152_T1_1mm_brain":
        mni_origin = [90, 126, 72]  
        # L and R are flipped between MNI and voxel space
        mni_coord = np.array([-1, 1, 1]) * mni_coord
        vox_coord = mni_coord + mni_origin

    elif template == "MNI152_T1_2mm_brain":
        mni_origin = [45, 63, 36]
        # L and R are flipped between MNI and voxel space
        mni_coord = np.array([-1, 1, 1]) * mni_coord/2
        vox_coord = mni_coord + mni_origin
    return vox_coord


def create_html_table(data, title="ROI results"):

    """Given a dictionary (keys as column names) returns a string that defines
    an html table."""

    # FIXME: This can be done more general obviously. This is a quick version
    # that give the results that we need for now.
    # When time: make the column names and values created on the 
    # call (and not hard coded as it is currently).
    html = """<TABLE BOARDER="5">
                <TR>
                    <TH COLSPAN="3">
                        <H3><BR>{table_title}</H3>
                    </TH>
                </TR>
                <TR>
                    <TH> name </TH>
                    <TH> xyz </TH>
                    <TH> mean </TH>
                    <TH> max </TH>
                </TR>
           """.format(table_title=title)

    table_content = ""
    for row in data:
        table_content += """<TR ALIGN="LEFT">
                                <TD> {name} </TD>
                                <TD> {xyz} </TD>
                                <TD> {mean} </TD>
                                <TH> {max} </TD>
                                <TH> {roi_sum} </TH>
                            </TR>
                        """.format(name=row["name"], xyz=row["xyz"], 
                                   mean=row["mean"], max=row["max"],
                                   roi_sum=row["roi_sum"])


    html += ("{content}\n</TABLE>".format(content=table_content))
    return html
