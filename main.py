from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Flatw0waCDM
from scipy.optimize import minimize

from scipy.stats import uniform
import numpy
from chainconsumer import Chain, ChainConsumer, make_sample
import pandas

def node_to_cosmo():
	n = 5000	
	cosmo_0 = FlatLambdaCDM(H0=70, Om0=0.3)

	w0_pdf = uniform(-1,1)
	wa_pdf = uniform(-0.2,0.4)
	Om0_pdf = uniform(cosmo_0.Om0-.1, cosmo_0.Om0+.1)

	zs = numpy.concatenate([numpy.arange(0.05,0.8,0.05),numpy.arange(0.8,1.21,0.1)])


	mu0=cosmo_0.distmod(zs)

	residuals = []

	for _ in range(n):
		_cosmo = Flatw0waCDM(H0=70, Om0=Om0_pdf.rvs(), w0 = w0_pdf.rvs(),wa = wa_pdf.rvs()) 
		mu = _cosmo.distmod(zs)
		residuals.append((mu - mu0).value)

	cols = ["N. {}".format(i) for i in range(len(zs))]
	df = pandas.DataFrame(residuals,columns=cols)

	c = ChainConsumer()
	c.add_chain(Chain(samples=df, name="Prior"))
	fig = c.plotter.plot(figsize=10)
	fig.show()

def cosmo_to_node():
	n = 1000

	cosmo_0 = FlatLambdaCDM(H0=70, Om0=0.3)

	zs = numpy.concatenate([numpy.arange(0.05,0.8,0.05),numpy.arange(0.8,1.21,0.1)])
	mu0=cosmo_0.distmod(zs).value

	def f(x, mu):
		_cosmo = Flatw0waCDM(H0=x[3], Om0=x[0], w0 = x[1],wa = x[2])
		ans = ((mu - _cosmo.distmod(zs).value)**2).sum()
		return ans

	results=[]
	for _ in range(n):
		mu = mu0 + uniform.rvs(loc=-.05,scale=.1,size=len(zs))
		res = minimize(f,(0.3,-1, 0, 70),args=mu,bounds=((0,1),(-5,1),(-10,10),(50,100)))
		if not res.success:
			raise Exception("Optimization Failed")
		results.append(res.x)

	ubf = {'$\Omega_M$':0.244, '$w_0$':-.735, '$w_a$':0  } 

	df = pandas.DataFrame(results,columns=[r'$\Omega_M$', r'$w_0$', r'$w_a$', r'$H_0$'])
	# df.to_csv("cosmo_to_model.tmp.csv")
	c = ChainConsumer()
	c.add_chain(Chain(samples=df, name="Prior"))
	c.add_marker(location=ubf,name="Union3 FlatwCDM",color='red',marker_size="40.")
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