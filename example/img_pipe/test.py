from img_pipe import img_pipe
import numpy as np
patient = img_pipe.freeCoG(subj='test_subj', hem='lh')
#patient.plot_recon_anatomy()

pial = patient.roi('pial',opacity=0.3, representation='surface',color=(0.9,0.8,0.5),gaussian=False)
hipp = patient.roi('lHipp', opacity = 0.8, representation = 'wireframe',color=(0.5, 0.3,0.5), gaussian=False)
#patient.plot_brain(rois=[pial, hipp])

#patient.check_pial()
#patient.convert_fsmesh2mlab()
#patient.convert_fsmesh2mlab(mesh_name ='inflated')
#patient.plot_brain()
#patient.get_subcort()
#subcort_roi = patient.roi(name='lPut',color=(1.0, 0.0,0.0))
#pial_roi = patient.roi (name ='lh_pial', opacity=0.5)
#patient.plot_brain(rois=[pial_roi,subcort_roi])

#patient.reg_img()

#patient.mark_electrodes()
#patient.project_electrodes(elecfile_prefix= 'hd_grid')
#grid_elecs = patient.get_elecs(elecfile_prefix='hd_grid')['elecmatrix']
#patient.plot_brain(elecs=grid_elecs)

patient.label_elecs(elecfile_prefix='elecs_all', atlas_surf='desikankilliany', atlas_depth='destrieux')
patient.warp_all(elecfile_prefix= 'elecs_all') # failed


subjs = ['test_subj',]
elecs = []
for s in subjs:
    print(s)
    patient = img_pipe.freeCoG(subj = s, hem = 'lh')
    elecs.append(patient.get_elecs(elecfile_prefix='TDT_elecs_all_warped')['elecmatrix'])
elecmatrix = np.concatenate(elecs, axis=0)
template = 'cvs_avg35_inMNI152'
atlas_patient = img_pipe.freeCoG(subj = template, hem='lh')
roi = atlas_patient.roi('pial', opacity=0.5)
atlas_patient.plot_brain(rois = [roi],
                         showfig=True,
                         screenshot=True,
                         elecs = elecmatrix,
                         weights = None)

patient.warp_all(elecfile_prefix = 'elecs_all')
patient.plot_recon_anatomy(elecfile_prefix = 'warped_elecs_file', template ='cvs_avg35_inMNI152')












