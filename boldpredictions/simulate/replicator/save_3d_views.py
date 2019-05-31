import os
import time
import numpy as np
import cortex

file_pattern = "{base}_{view}_{surface}.png"

# code golf version
# combines two dicts in one nasty statement!
_combine = lambda a,b: ( lambda c: [c, c.update(b)][0] )(dict(a))

# normal, boring-ass version
# def _combine(a, b):
#     c = dict(a)
#     c.update(b)
#     return c

_tolists = lambda p: {k:[v] for k,v in p.items()}

def save_3d_views(data, root, path_base, size=(1024, 768), trim=True,
                  view_names = ["lateral", "front", "back", "top", "bottom", "medial"]):
    """Saves 3D views of `data` in and around `root`.
    """
    # Create viewer
    handle = cortex.webshow(data)

    # Wait until it's up or something?
    # (this is fucking stupid
    # there should be some way to see that it's open
    # but I don't know what it is)
    time.sleep(5.0)
    
    # Set up params
    basic = dict(projection=['orthographic'], radius=260)
    
    views = dict(lateral=dict(altitude=90.5, azimuth=181, pivot=180),
                 medial=dict(altitude=90.5, azimuth=0, pivot=180),
                 front=dict(altitude=90.5, azimuth=0, pivot=0),
                 back=dict(altitude=90.5, azimuth=181, pivot=0),
                 top=dict(altitude=0, azimuth=180, pivot=0),
                 bottom=dict(altitude=180, azimuth=0, pivot=0))
    views = dict([(s,views[s]) for s in view_names])

    surfaces = dict(inflated=dict(mix=0.5),
                    fiducial=dict(mix=0.0))

    # Save views!
    filenames = dict([(key, {}) for key in surfaces.keys()])
    # filenames = []
    for view,vparams in views.items():
        for surf,sparams in surfaces.items():
            # Combine basic, view, and surface parameters
            params = _combine(_combine(basic, vparams), sparams)

            # Set the view
            handle._set_view(**_tolists(params))

            # Save image, store filename
            filename = file_pattern.format(base=path_base, view=view, surface=surf)
            filenames[surf][view] = filename
            # filenames.append(filename)

            output_path = os.path.join(root, filename)
            handle.saveIMG(output_path, size)

            # Trim edges?
            if trim:
                # Wait for browser to dump file
                while not os.path.exists(output_path):
                    pass
                
                time.sleep(0.5)

                try:
                    import subprocess
                    subprocess.call(["convert", "-trim", output_path, output_path])
                except:
                    pass

    # Close the window!
    try:
        handle.close()
        return filenames
    except:
        return filenames



if __name__ == "__main__":
    # Test this shit out
    data = cortex.Volume.random("S1", "fullhead")
    filenames = save_3d_views(data, "", "test")
