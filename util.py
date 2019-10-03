def time_transform(t):
	return (2./365*t)

def inv_time_transform(t):
	return (365./2*t)

def period():
	return(time_transform(365)-time_transform(0))
	