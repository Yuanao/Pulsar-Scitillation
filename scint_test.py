import scint as sci

s=sci.Sci()
#dyn ,dt,df, freq = s.load_file('combine_J1518+4904.ar.ds')
#s.load_file('J0055+5117-b2048.rf.ds',fmin=1050,fmax=1200)
#s.load_file('test_30.rf.PP.ds',fmin=1050,fmax=1200)
s.load_file('J1518+4904_0704-15.ds',fmin=1050,fmax=1200)
#s.pro_dyn(model='wavelet')
s.get_sspec()
s.find_sym(plot=False,f_stat=-3,f_end=3)
#s.find_sym(plot=True,f_stat=-10,f_end=10)
#s.find_sym()
#s.arc_trans()
#s.arc_trans(eta_step=2000,load_power=False,plot_model=True,plot_power=True,
#            eta_min=0.02,eta_max=1,kendalltau=False)
s.arc_para(eta_step=1000,load_power=True,plot_model=False,plot_power=True,eta_min=0.02,eta_max=1,kendalltau=False)
#s.fit_arc(plot=True,pick=[0])
#s.fit_arc(plot=True,pick=[0.02,0.063,0.082,0.25972986, 0.38915458, 0.51367684])
s.fit_arc(plot=True,pick=[0])
#s.fit_arc(plot=True)
s.plot_dyn()
#s.plot_sspec(plot_arc=False)
s.plot_sspec()

