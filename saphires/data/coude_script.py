import saphires as saph
import pickle as pkl
import numpy as np

multi_spec = pkl.load(open('HD283818_Coude_py2.pkl','rb'))

temp_spec = saph.io.read_pkl('lte06000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes_2800-11000_air.pkl',temp=True)

tar = np.empty(0)
tar_spec = {}

for i in range(multi_spec['wav'].shape[0]):
	tar_i,tar_spec_i = saph.io.read_vars(multi_spec['wav'][i],multi_spec['flux'][i],'HD283818_'+np.str(i+1),w_file='coude.ls',combine_all=False)
	tar = np.append(tar,tar_i)
	tar_spec.update(tar_spec_i)

tar_spec = saph.utils.apply_shift(tar,tar_spec,0)
tar_spec = saph.utils.prepare(tar,tar_spec,temp_spec,cr_trim=-0.3,oversample=1)

tar_spec = saph.bf.compute(tar,tar_spec,vel_width=400)
tar_spec = saph.bf.analysis(tar,tar_spec,sb='sb1',R=40000,single_plot=True,fit_trim=20)

