from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Flatw0waCDM
from astropy.cosmology import Planck15
from scipy.optimize import minimize

from scipy.stats import uniform
import numpy
from chainconsumer import Chain, ChainConsumer, make_sample
import pandas
import matplotlib.pyplot as plt

import jax
import jax_cosmo as jc
import jax.numpy as jnp
import astropy
from astropy.io import fits
jax.config.update("jax_enable_x64", True)

def pdf():

	# f2="mu_mat_union3_cosmo=2_mu.fits"
	# hdu_list2 = fits.open(f2, memmap=True)
	f1="mu_mat_union3_cosmo=2.fits"
	hdu_list = fits.open(f1, memmap=True)
	invcov = hdu_list[0].data
	invcov = invcov.astype("float")
	zs = invcov[0,1:]
	n0 = invcov[1:,0]
	invcov = invcov[1:,1:]

	aas = 1/(1+zs)

	# w0s = numpy.linspace(-1.2,-0.2,3)
	# was =  numpy.linspace(-2, 2 ,3)
	w0s = numpy.linspace(-1.1,-0.3,12)
	was =  numpy.linspace(-3, 1, 12)	
	Om0s = numpy.linspace(0.3-.05, 0.3+.05,9)

	cosmo_0 = jc.Planck15(Omega_c = 0.3, Omega_b=0, w0=-1., wa=0.)
	dL = (1+zs) * jc.background.transverse_comoving_distance(cosmo_0 ,aas) # In [Mpc/h]
	mu_0 = 5*jnp.log10(dL) + 25

	def nodes(W):
		cosmo = jc.Planck15(Omega_c = W[0],Omega_b=0,  w0=W[1], wa=W[2])
		dL = (1+zs) * jc.background.transverse_comoving_distance(cosmo ,aas) # In [Mpc/h]
		mu = 5*jnp.log10(dL) + 25
		nodes = mu-mu_0
		return nodes

	J_nodes = jax.jacfwd(nodes)

	omega=numpy.zeros((len(Om0s),len(w0s),len(was)))
	lnp_union = numpy.zeros((len(Om0s),len(w0s),len(was)))
	# omega = dict()
	for i, Om0 in enumerate(Om0s):
		for j, w0 in enumerate(w0s):
			for k, wa in enumerate(was):
				W = jnp.array((Om0, w0, wa))
				N = nodes(W)
				J = J_nodes(W)
				omega[i,j,k] = jnp.sqrt(jnp.linalg.det(jnp.dot(J.T,J)))
				lnp_union[i,j,k] = -0.5 * ((N.T-n0) @ invcov @ (N.T-n0))

	levels = np.arange(-18, 0, 1)
	fig, axs = plt.subplots(3,3, figsize=(12,12))
	for Om0s_index, ax in enumerate(axs.flat):
		CS = ax.contour(w0s, was, lnp_union[Om0s_index,:,:].T, levels=levels, colors='red',label=r'$\ln{p}$')
		ax.clabel(CS, levels, inline=True, fontsize=10)
		CS2 = ax.contour(w0s, was, lnp_union[Om0s_index,:,:].T+ numpy.log(omega[Om0s_index,:,:].T), levels=levels,colors='blue',label=r'$\ln{p} + \ln{w}$')
		ax.clabel(CS2, levels, inline=True, fontsize=10)
		ax.set_xlabel(r"$w_0$")
		ax.set_ylabel(r"$w_a$")
		ax.set_title("$\Omega_M={:7.5f}$".format(Om0s[Om0s_index]))

	fig.suptitle(r"$\ln{p}$ (red); $\ln{p}+\ln{w}$ (blue)")
	fig.tight_layout()	
	fig.savefig('contour.png')
	fig.show()


	fig, axs = plt.subplots(3,3, figsize=(9,9))
	for Om0s_index, ax in enumerate(axs.flat):
		CS = ax.contour(w0s, was, numpy.log(omega[Om0s_index,:,:].T), colors='red',label=r'$\ln{p}$')
		ax.clabel(CS, CS.levels, inline=True, fontsize=10)
		ax.set_title("$\Omega_M={:7.5f}$".format(Om0s[Om0s_index]))
		ax.set_xlabel(r"$w_0$")
		ax.set_ylabel(r"$w_a$")
		fig.colorbar(im, ax=ax)
	fig.suptitle(r"$\ln{w}$")
	fig.tight_layout()	
	fig.savefig('result.png')

	plt.show()




def node_to_cosmo():
	cosmo_desi = Flatw0waCDM(H0=70, Om0=0.343, w0 = -0.64, wa = -1.27)
	cosmo_0 = FlatLambdaCDM(H0=70, Om0=0.3)

	n = 10000

	w0_pdf = uniform(cosmo_desi.w0-2*0.11, 4*0.11)
	wa_pdf = uniform(cosmo_desi.wa-2*0.37, 4*0.37)
	Om0_pdf = uniform(cosmo_0.Om0-.05, cosmo_0.Om0+.1)

	zs = numpy.concatenate([numpy.arange(0.05,0.8,0.05),numpy.arange(0.8,1.21,0.1)])

	mu0=cosmo_0.distmod(zs)

	residuals_Om0 = []
	residuals_w0 = []
	residuals_wa = []

	for _ in range(n):
		_cosmo = Flatw0waCDM(H0=70, Om0=Om0_pdf.rvs(), w0 = -1,wa = 0)
		mu = _cosmo.distmod(zs)
		residuals_Om0.append((mu - mu0).value)

		_cosmo = Flatw0waCDM(H0=70, Om0=cosmo_0.Om0, w0 = w0_pdf.rvs(), wa = 0)
		mu = _cosmo.distmod(zs)
		residuals_w0.append((mu - mu0).value)

		_cosmo = Flatw0waCDM(H0=70, Om0=cosmo_0.Om0, w0 = -1, wa = wa_pdf.rvs())
		mu = _cosmo.distmod(zs)
		residuals_wa.append((mu - mu0).value)	

	residuals_Om0 = numpy.array(residuals_Om0)
	residuals_w0 = numpy.array(residuals_w0)
	residuals_wa = numpy.array(residuals_wa)

	cols = ["N. {}".format(i) for i in range(len(zs))]
	df_Om0 = pandas.DataFrame(residuals_Om0,columns=cols)
	df_w0 = pandas.DataFrame(residuals_w0,columns=cols)
	df_wa = pandas.DataFrame(residuals_wa,columns=cols)

	fig, axs = plt.subplots(4,5, figsize=(12, 8), layout='constrained')
	for index, ax in enumerate(axs.flat):
	    ax.set_title("Index {}".format(index))
	    ax.hist(
	    	[df_Om0.iloc[:,index],df_w0.iloc[:,index],df_wa.iloc[:,index]], 20,
	    	label=[r"$\Omega_M$",r"$w_0$",r"$w_a$"]
	    	)
	    # ax.hist(df_w0.iloc[:,index],label=r"$w_0$")
	    # ax.hist(df_wa.iloc[:,index],label=r"$w_a$")
	    if index==0:
	    	ax.legend()

	fig.show()

	fig, axs = plt.subplots(4,5, figsize=(12, 8), layout='constrained')
	for index, ax in enumerate(axs.flat):
	    ax.set_title("Index {}".format(index))
	    ax.hist(
	    	[df_w0.iloc[:,index],df_wa.iloc[:,index]], 20,
	    	label=[r"$w_0$",r"$w_a$"]
	    	)
	    if index==0:
	    	ax.legend()

	fig.show()

	fig, axs = plt.subplots(4,5, figsize=(12, 8), layout='constrained')
	for index, ax in enumerate(axs.flat):
	    ax.set_title("Index {}".format(index))
	    ax.hist(df_w0.iloc[:,index],label=r"$w_0$")

	fig.show()


	c = ChainConsumer()
	# c.add_chain(Chain(samples=df_Om0, name=r"$\Omega_M$"))
	c.add_chain(Chain(samples=df_w0, name=r"$w_0$"))
	c.add_chain(Chain(samples=df_wa, name=r"$w_a$"))

	fig = c.plotter.plot(figsize=10)
	fig.show()

def cosmo_to_node():
	n = 100

	cosmo_0 = FlatLambdaCDM(H0=70, Om0=0.3)

	zs = numpy.concatenate([numpy.arange(0.05,0.8,0.05),numpy.arange(0.8,1.21,0.1)])
	mu0=cosmo_0.distmod(zs).value

	def f(x, mu):
		_cosmo = Flatw0waCDM(H0=x[3], Om0=x[0], w0 = x[1],wa = x[2])
		ans = ((mu - _cosmo.distmod(zs).value)**2).sum()
		return ans

	index=0
	results=[]
	while (index<n):
		mu = mu0 + uniform.rvs(loc=-.1,scale=.2,size=len(zs))
		res = minimize(f,(0.3,-1, 0, 70),args=mu,bounds=((0,1),(-5,1),(-10,10),(50,100)))
		if res.success and res.fun<0.04:
			print(res.fun)
			results.append(res.x)
			index = index+1


	df = pandas.DataFrame(results,columns=[r'$\Omega_M$', r'$w_0$', r'$w_a$', r'$H_0$'])
	# df.to_csv("cosmo_to_model.tmp.csv")
	c = ChainConsumer()
	c.add_chain(Chain(samples=df, name="Prior"))
	ubf = {'$\Omega_M$':0.244, '$w_0$':-.735, '$w_a$':0  } 
	sc = {'$w_0$':-1, '$w_a$':0  } 

	c.add_marker(location=ubf,name="Union3 FlatwCDM",color='red',marker_size="40.")
	c.add_marker(location=sc,name="LCDM",color='green',marker_size="40.")

	fig = c.plotter.plot(figsize=10)
	fig.show()

cosmo_to_node()


	# residuals = []

	# for _ in range(n):
	# 	_cosmo = Flatw0waCDM(H0=70, Om0=Om0_pdf.rvs(), w0 = w0_pdf.rvs(),wa = wa_pdf.rvs()) 
	# 	mu = _cosmo.distmod(zs)
	# 	residuals.append((mu - mu0).value)

	# cols = ["N. {}".format(i) for i in range(len(zs))]
	# df = pandas.DataFrame(residuals,columns=cols)

	c = ChainConsumer()
	c.add_chain(Chain(samples=df, name="Prior"))
	fig = c.plotter.plot(figsize=10)
	fig.show()