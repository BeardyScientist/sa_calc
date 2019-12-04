# TO DO LIST
# *******************

# ESSENTIAL

# DESIRABLE
# Make into an excecutable

# NICE TO HAVE
# Make so it can accept a list of percentile values and will plot them each over each other.
# Size gating functionality
# Make it so you can pass arguments straight from terminal 
# Make it into a library people can import into their own code
# Put minor functions into the “processed_data()” function





# DEPENDENCIES 

import os
import platform
import sys
import re
import time
import pickle
import numpy as numpty
import pandas
import matplotlib.patheffects as PathEffects
import matplotlib.transforms as transforms






# FUNCTIONS

def clear():
	"""Clears the screen in a cross-platform-friendly manner
	
	"""

	if platform.system() == "Windows":
		os.system('cls')
	else:
		os.system('clear')


def sort_data(data):
	"""Sorts the data by angle in ascending order.
	
	Parameters
	----------
	data: np.ndarray
		The data to be sorted.
		
	Returns
	-------
	sorted_data: np.ndarray
		The data sorted by angle in ascending order.
	"""

	sorted_data = data[data[:,0].argsort()]
	
	return sorted_data





def make_bidirectional(data):
	"""Takes directional data from 0 – 180 degrees and duplicates it about 180 degrees.
	
	Parameters
	----------
	data: np.ndarray
		The data to be made bidirectional.
		
	Returns
	-------
	data: np.ndarray
		The data duplicated about 180 degrees.
	"""


	length = len(data)
	i = 0
	while i < length:	
		data = numpty.append(data, [[data[i][0] + 180, data[i][1]]], axis=0)
		i += 1		
	
	return data
	
	
	
	
	
def make_cardinal(data): 
	"""Makes orientation data increase clockwise from the vertical position, as in cardinal data.
	
	Takes directional data ascending anti-clockwise with zero on the right 
	(as produced by ImageJ and Adobe Illustrator) and converts into data
	ascending clockwise with zero at the top (as in a compass).
	
	Parameters
	----------
	data: np.ndarray
		The data to be made cardinal.
		
	Returns
	-------
	data: np.ndarray
		The data rotated by 90 degrees and with the direction inverted.
	"""
	
	length = len(data)
	i = 0
	while i < length:
	
		if data[i][0] > 90:
			data[i][0] = (360 - data[i][0]) + 90
		else:
			data[i][0] = ((360 - data[i][0]) + 90) - 360
				
		i += 1
	
	return data		





def average_by_angle(data):

	"""Calculates the average aspect-ratio for each degree of orientation.
	
	First, this function floors the data down to the nearest integer.
	Second, it builds a new array with angles from 0 to 359 and populates the 
	aspect-ratios with the average aspect-ratio from the input data that
	shares the same floored angle. 
	
	Parameters
	----------
	data: np.ndarray
		The data to be averaged by angle.
		
	Returns
	-------
	avg_data: np.ndarray
		An array of angles from 0 to 359 and aspect-ratios averaged from the input data.
	"""
	
	# Floors the angle data
	floored_data = data.copy()
	length = len(floored_data)
	i = 0
	while i < length:	
		floored_data[i][0] = numpty.floor(floored_data[i][0])
		i += 1	
	
	# Finds the average AR value for each value of floor
		
	temp_array = numpty.array([floored_data[0][1]])

	avg_data = numpty.empty([360,2])
	for x in range(0, 360):
		avg_data[x][0] = x
		avg_data[x][1] = numpty.NaN

	length = len(floored_data)
	i = 1
	
	while i < length:
		if floored_data[i][0] == floored_data[i-1][0]:
			temp_array = numpty.append(temp_array, [floored_data[i][1]])
		else:
			angle = int(floored_data[i-1][0])
			avg_data[angle][1] = numpty.mean(temp_array)
	
			temp_array = numpty.array([floored_data[i][1]])
		i += 1

	angle = int(floored_data[i-1][0])
	avg_data[angle][1] = numpty.mean(temp_array)
	
	return avg_data





def calculate_detrend_by(data): 

	"""Calculates the value by which the dataset should be detrended by.
	
	This function creates a list containing the magnitude of the trend (calculated 
	using the find_trend_magnitude() function) for each degree of orientation the 
	dataset is rotated by, then returns the index at which the minimum trend 
	magnitude can be found.
	
	
	Parameters
	----------
	data: np.ndarray
		The input data from which the detrend_by angle will be calculated.
		
	Returns
	-------
	angle_to_detrend_by: int
		The angle by which the dataset should be rotated to detrend it.
	"""

	def find_trend_magnitude(data):
	
		"""Finds the magnitude of the trend in the dataset.
			
		This function firsts calculates X/Y coordinates for each datapoint, then removes
		any data from the lower 2 quadrants (as this will be a duplicate of the data from
		the upper 2 quadrants), then it separates the data from the top-left and top-right
		quadrants. It then takes subtracts the x-axis spread from the top-left quadrant
		from the x-axis spread from the top-right quadrant and returns the difference. 
	
	
		Parameters
		----------
		data: np.ndarray
			The input data from which the trend magnitude will be calculated.
		
		Returns
		-------
		trend_magnitude: float
			The magnitude of the trend in the dataset at this degree of rotation.
		"""
		
		# Calculates X/Y coordinates for each datapoint
	
		length = len(data)
		i = 0

		xy_data = numpty.array([[0,0],
								[0,0]])

		while i < length:
			if data[i][1] > 0:
				x = data[i][1] * numpty.sin(data[i][0] * 0.0174533)
				y = data[i][1] * numpty.cos(data[i][0] * 0.0174533)

				xy_data = numpty.append(xy_data, [[x,y]], axis=0)
			i += 1
		
		# Removes datapoints where y < 0	
		
		length = len(xy_data)
		i = 0
		death_note = []

		for i in range(0, length):
			if xy_data[i][1] < 0:
				death_note = death_note + [i]

		filtered_out_y_below_zero = numpty.delete(xy_data, death_note, axis=0)		
		
		# Creates array where x > 0

		length = len(filtered_out_y_below_zero )
		i = 0
		death_note = []

		for i in range(0, length):
			if filtered_out_y_below_zero[i][0] < 0:
				death_note = death_note + [i]

		x_greater_than_zero = numpty.delete(filtered_out_y_below_zero, death_note, axis=0)
		x_greater_than_zero = x_greater_than_zero[:,0]
		sum_x_greater_than_zero	= numpty.sum(x_greater_than_zero)
		
		# Creates array where x < 0

		length = len(filtered_out_y_below_zero )
		i = 0
		death_note = []

		for i in range(0, length):
			if filtered_out_y_below_zero[i][0] > 0:
				death_note = death_note + [i]

		x_less_than_zero = numpty.delete(filtered_out_y_below_zero, death_note, axis=0)
		x_less_than_zero = x_less_than_zero[:,0]
		sum_x_less_than_zero = numpty.sum(x_less_than_zero)
	
		# Finds difference between sums of x>0 and x<0
	
		trend_magnitude = (sum_x_greater_than_zero + 100) - (sum_x_less_than_zero + 100)
		
		return trend_magnitude
		
		
	trend_magnitudes_at_all_angles = [find_trend_magnitude(data)]
	
	for x in range(0,360):

		# Shifts data by 1 each time loop is completed
		length = len(data)
		i = 0
		while i < length:
			data[i][0] = data[i][0] + 1
			if data[i][0] > 359:
				data[i][0] = data[i][0] - 360
			i += 1

		trend_magnitudes_at_all_angles = trend_magnitudes_at_all_angles + [find_trend_magnitude(data)]
	
	trend_magnitudes_at_all_angles_array = numpty.array(trend_magnitudes_at_all_angles)
	angle_to_detrend_by = numpty.argmin(trend_magnitudes_at_all_angles)
		
	return angle_to_detrend_by





def detrend(data, angle_to_detrend_by): # Detrends the dataset

	"""Detrends the data by the amount specified in the calculate_detrend_by() function

	Parameters
	----------
	data: np.ndarray
		The input data that will be detrended.
	angle_to_detrend_by: int
		The angle by which the data will be rotated.
	
	Returns
	-------
	data: np.ndarray
		The detrended data.
	"""

	length = len(data)
	i = 0
	while i < length:
		data[i][0] = data[i][0] + angle_to_detrend_by
		if data[i][0] > 359:
			data[i][0] = data[i][0] - 360
		if data[i][0] < 1:
			data[i][0] = data[i][0] + 359
		i += 1
	
	return data





def retrend(data, retrend_by):

	"""Retrends the detrended dataset to its orientation in cardinal space

	Parameters
	----------
	data: np.ndarray
		The input data that will be retrended.
	retrend_by: int
		The angle by which the data will be rotated.
	
	Returns
	-------
	data: np.ndarray
		The detrended data.
	"""

	length = len(data)
	i = 0
	while i < length:
		data[i][0] = data[i][0] - retrend_by
		if data[i][0] > 359:
			data[i][0] = data[i][0] - 360
		if data[i][0] < 1:
			data[i][0] = data[i][0] + 359
		i += 1
	
	return data





def calculate_ellipse_axes(data, percentile): # Calculates the axes of the structural anisotropy ellipse

	"""Calculates the axes of the structural anisotropy ellipse
	
	This function takes detrended_data as an input, calculates the X/Y coordinates
	for each dataset, then calculates the specified percentile on both the x and y axes.
	
	Ellipses are calculated at a percentile to limit the effect of outliers. A suitable
	percentile value should be chosen on the basis of the number of datapoints in the
	dataset, although SA values calculated at different percentiles should not be 
	compared against each other. 95 is the suggested percentile if there are 10s of 
	datapoints, 99 is the suggested percentile if there are 100s – 1000s of datapoints.

	Parameters
	----------
	data: np.ndarray
		The input data that will be retrended.
	percentile: int
		The percentile at which the SA value will be calculated.
	
	Returns
	-------
	ellipse_axes: list
		List containing the long axis of the ellipse, followed by the short axis
	"""
	
	# Calculates X/Y coordinates for each datapoint

	length = len(data)
	i = 0

	x_data = numpty.array([[0,0],
							[0,0]])
							
	y_data = numpty.array([[0,0],
							[0,0]])

	while i < length:
		if data[i][1] > 0:
			x = data[i][1] * numpty.sin(data[i][0] * 0.0174533)
			y = data[i][1] * numpty.cos(data[i][0] * 0.0174533)

			x_data = numpty.append(x_data, [x])
			y_data = numpty.append(y_data, [y])
		i += 1	
	
	short_axis = numpty.percentile(x_data, percentile)
	long_axis = numpty.percentile(y_data, percentile)
	
	ellipse_axes = [long_axis, short_axis]
	
	return ellipse_axes





def convert_to_xy(data): # Converts radial data into xy data
	
	"""Converts radial data into xy data

	Parameters
	----------
	data: np.ndarray
		The input data to be converted into X/Y data.
	
	Returns
	-------
	xy_data: np.ndarray
		X/Y data converted from the radial input data
	"""
	
	# Calculates X/Y coordinates for each datapoint

	length = len(data)
	i = 0

	xy_data = numpty.array([[0,0],
							[0,0]])

	while i < length:
		if data[i][1] > 0:
			x = data[i][1] * numpty.sin(data[i][0] * 0.0174533)
			y = data[i][1] * numpty.cos(data[i][0] * 0.0174533)

			xy_data = numpty.append(xy_data, [[x,y]], axis=0)
		i += 1
		
	xy_data = numpty.delete(xy_data, [0,1], axis=0)
	
	return xy_data





def gate(data, axis_max): 

	"""Removes data outside of the axis_max
	
	This function strips out any data points that exceed the value of axis_max. 
	This is intended to allow plots to be generated of data which contains values
	exceeding the axis_max (such as outliers) while allowing the SA calculations to 
	be performed on the whole dataset. This function is only intended to be used for 
	plotting data, and is not appropriate for calculations of the SA value. 

	Parameters
	----------
	data: np.ndarray
		The input data to be gated
	axis_max: int
		
	
	Returns
	-------
	gated_data: np.ndarray
		The dataset minus any datapoints whose AR values exceed axis_max
	"""

	length = len(data)
	i = 0
	death_note = []

	for i in range(0, length):
		if data[i][1] > axis_max:
			death_note = death_note + [i]
		elif data[i][1] < -axis_max:
			death_note = death_note + [i]

	gated_data = numpty.delete(data, death_note, axis=0)	
	
	return gated_data
	
	
	
	
def process_data(input_file, percentile, axis_max, photo_orientation):

	"""Receives the input values and calls the other functions 
	
	This function strips out any data points that exceed the value of axis_max. 
	This is intended to allow plots to be generated of data which contains values
	exceeding the axis_max (such as outliers) while allowing the SA calculations to 
	be performed on the whole dataset. This function is only intended to be used for 
	plotting data, and is not appropriate for calculations of the SA value. 

	Parameters
	----------
	input_file: str
		The name of a CSV file containing the input data, formatted with angle in 
		the first column and aspect ratio in the second column. File extension 
		is required.
	percentile: int
		The percentile at which the SA value will be calculated.
	axis_max: int
		The maximum value of the axis on the SA plot.
	photo_orientation: int
		The angle to which north can be found in the input photograph.
		
	
	Returns
	-------
	detrended_data
		The input dataset that has been sorted, made bidirectional, made cardinal, and
		been detrended.
	detrended_xy_data: np.ndarray
		The detrended dataset converted to X/Y coordinates.
	retrended_data: np.ndarray
		The dataset retrended into its cardinal orientation
	retrended_xy_data: np.ndarray
		The retrended dataset converted to X/Y coordinates.
	rounded_structural_anisotropy: float
		The structural anisotropy value rounded to two significant figures.
	retrend_by: int
		The angle by which the detrended data must be rotated to be retrended into its
		cardinal orientation.
	trend: int
		The trend in the dataset with respect to north.
	ellipse_axes: list
		The long and short axes of the structural anisotropy ellipse.
	"""

	# Validate photo orientation

	if photo_orientation == "none":
		angle_to_north = 0	
	elif photo_orientation.isdigit() == True:
		angle_to_north = int(photo_orientation)
	else:
		raise Exception("Invalid orientation input")


	# Input Data		
	
	f = open(input_file, "r+")
	f.close()
	
	input_data = numpty.array(pandas.read_csv(input_file, header = None))

	if numpty.isnan(input_data).any() == True:
		raise Exception("NaNs in dataset.")


	# Process Data

	sorted_data = sort_data(input_data)

	if numpty.floor(numpty.max(input_data[:,0])) <= 179:
		bidirectional_data = make_bidirectional(sorted_data)
	elif numpty.floor(numpty.max(input_data[:,0])) > 179 and numpty.floor(numpty.max(input_data[:,0])) <= 359:
		bidirectional_data = sorted_data
	else:
		raise Exception("Maximum angle exceeds 360.")
		
	working_data = make_cardinal(bidirectional_data)

	detrend_by = calculate_detrend_by(average_by_angle(working_data))
	detrended_data = detrend(working_data, detrend_by)
	detrended_xy_data = convert_to_xy(detrended_data)
	ellipse_axes = calculate_ellipse_axes(detrended_data, percentile)

	structural_anisotropy = ellipse_axes[0] / ellipse_axes[1]
	rounded_structural_anisotropy = numpty.round(structural_anisotropy, 1)

	if photo_orientation == "none":
		retrend_by = 0
	elif  detrend_by + angle_to_north >= 0:
		retrend_by = detrend_by + angle_to_north
	elif  detrend_by + angle_to_north < 0:
		retrend_by = detrend_by + angle_to_north + 180
	if  detrend_by + angle_to_north > 180:
		retrend_by = detrend_by + angle_to_north - 180
		
	if retrend_by >= 0 and retrend_by < 180:
		trend = 360 - retrend_by - 180
	elif retrend_by < 0:
		trend = 360 - retrend_by + 180
	elif retrend_by > 180:
		trend = 360 - retrend_by

	retrended_data = retrend(detrended_data, retrend_by)
	retrended_xy_data = convert_to_xy(retrended_data)
	
	
	return [detrended_data, detrended_xy_data, retrended_data, retrended_xy_data, rounded_structural_anisotropy, retrend_by, trend, ellipse_axes]





def draw_plot(action, filename): # Draws the structural anisotropy diagram

	"""Draws the structural anisotropy diagram
	
	This function draws the structural anisotropy diagram, either for previewing within 
	the python viewer, or for saving to a file. 
	
	
	NOTE: Outputs from the process_data() function must be available to this function.

	Parameters
	----------
	action: str
		The action the function should take. Either “preview”, “save_raster”, 
		“save_vector”, or “save_vector_svg”.
	filename: str
		The filename of the file this function should save.
	"""

	import matplotlib.pyplot as pyplot

	if photo_orientation == "none":
		gated_data = convert_to_xy(gate(detrended_data, axis_max))
	elif photo_orientation.isdigit() == True:
		gated_data = convert_to_xy(gate(retrended_data, axis_max))
		
	

	rotated_data = numpty.rot90(gated_data)

	rotate = transforms.Affine2D().rotate_deg(retrend_by)
	base = pyplot.gca().transData

	fig = pyplot.figure(1, figsize=(5, 5))
	
	file_name_regex = re.findall(".*/(.*)$", input_file)
	if (file_name_regex):
	  input_file_name = file_name_regex[0]
	else:
	  input_file_name = input_file
		  
	fig.suptitle("Structural Anisotropy: " + input_file_name)
	
	theta = numpty.linspace(-numpty.pi, numpty.pi, 200)
	
	if photo_orientation.isdigit() == True:
		north_label = pyplot.text(0, axis_max + 1, "N", color="#aaaaaa", ha="center", va="center")
		south_label = pyplot.text(0, -axis_max - 0.75, "S", color="#aaaaaa", ha="center", va="center")
		east_label = pyplot.text(axis_max + 0.75, 0, "E", color="#aaaaaa", ha="center", va="center")
		south_label = pyplot.text(-axis_max - 0.75, 0, "W", color="#aaaaaa", ha="center", va="center")
		east_line = pyplot.plot([1, axis_max], [0, 0], "-", color="#aaaaaa", linewidth=0.5)
		west_line = pyplot.plot([-axis_max, -1], [0, 0], "-", color="#aaaaaa", linewidth=0.5)
		north_line = pyplot.plot([0, 0], [1, axis_max], "-", color="#aaaaaa", linewidth=0.5)
		south_line = pyplot.plot([0, 0], [-1, -axis_max], "-", color="#aaaaaa", linewidth=0.5)

	for x in range(0, axis_max + 1):
		pyplot.plot(x * numpty.sin(theta), x * numpty.cos(theta), color="#aaaaaa", linewidth=0.5)
		if x != 0:
			axis_label = pyplot.text(0, x - 0, x, color="#5e5e5e", ha="center", va="center")
			if axis_max <= 10:
				axis_label.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
				axis_label.set_fontsize(10)
			else:
				axis_label.set_path_effects([PathEffects.withStroke(linewidth=1 / axis_max * 40, foreground='w')])
				axis_label.set_fontsize(1 / axis_max * 100)
			
	if photo_orientation == "none":
		points = pyplot.plot(rotated_data[1], rotated_data[0], ".", color="#000000", markersize=3)
		ellipse = pyplot.plot(ellipse_axes[1] * numpty.sin(theta), ellipse_axes[0] * numpty.cos(theta), color="#EB234B", linewidth=3)
		pyplot.annotate("SA" + str(percentile) + " = " + str(rounded_structural_anisotropy), xy=(0.85, 0.95), xycoords='axes fraction', ha="left", va="center", bbox=dict(facecolor='white', alpha=1))
	if photo_orientation.isdigit() == True:
		points = pyplot.plot(rotated_data[1], rotated_data[0], ".", color="#000000", markersize=3)
		ellipse = pyplot.plot(ellipse_axes[1] * numpty.sin(theta), ellipse_axes[0] * numpty.cos(theta), color="#EB234B", linewidth=3, transform=rotate + base)
		pyplot.annotate("SA" + str(percentile) + " = " + str(rounded_structural_anisotropy) + "\nTrend: " + str(trend) + "°", xy=(0.85, 0.95), xycoords='axes fraction', ha="left", va="center", bbox=dict(facecolor='white', alpha=1))


	pyplot.axis("scaled")
	pyplot.axis([-axis_max - 2, axis_max + 2, -axis_max - 2, axis_max + 2])
	pyplot.axis("off")

	if action == "preview":
		pyplot.show()
	elif action == "save_raster":
		pyplot.savefig(filename + ".png")
	elif action == "save_vector":
		pyplot.savefig(filename + ".pdf")
	elif action == "save_vector_svg":
		pyplot.savefig(filename + ".svg")
		
	pyplot.clf()
	
	return





def load_settings(): 

	"""Loads saved settings from the preferences file or, if no file exists, assigns default settings
	
	Returns
	-------
	input_file: str
		The name of a CSV file containing the input data, formatted with angle in 
		the first column and aspect ratio in the second column. File extension 
		is required.
	percentile: int
		The percentile at which the SA value will be calculated.
	axis_max: int
		The maximum value of the axis on the SA plot.
	photo_orientation: int
		The angle to which north can be found in the input photograph.
	"""

	if os.path.exists("sa_calc_prefs") == True:
	
		settings_file = open("sa_calc_prefs", "rb")
		settings = pickle.load(settings_file)

		input_file = settings[0]
		percentile = settings[1]
		axis_max = settings[2]
		photo_orientation = settings[3]
	else:
		input_file = "data1.csv"
		percentile = 95
		axis_max = 10
		photo_orientation = "none"
		
	return [input_file, percentile, axis_max, photo_orientation]





def save_settings(input_file, percentile, axis_max, photo_orientation): 

	"""Saves settings from this session to be loaded on the next session
	
	Parameters
	----------
	input_file: str
		The name of a CSV file containing the input data, formatted with angle in 
		the first column and aspect ratio in the second column. File extension 
		is required.
	percentile: int
		The percentile at which the SA value will be calculated.
	axis_max: int
		The maximum value of the axis on the SA plot.
	photo_orientation: int
		The angle to which north can be found in the input photograph.
	"""
	
	settings = [input_file, percentile, axis_max, photo_orientation]
	settings_file = open("sa_calc_prefs", "wb")
	pickle.dump(settings, settings_file)
	settings_file.close()

	return





# USER INTERFACE

help_text = "\n\nFor instructions on how to use the Structural Anisotropy Calculator,\nplease refer to the step-by-step guide found at XXXXXX.com.\n"

# Starting variables

input_file = load_settings()[0]
percentile = load_settings()[1]
axis_max = load_settings()[2]
photo_orientation = load_settings()[3]

proceed_lvl_zero = False
while proceed_lvl_zero == False: # Master UI loop
	
	wait_for_file = False
	while wait_for_file == False:
		clear()
		print("STRUCTURAL ANISOTROPY CALCULATOR v1\n***********************************\n")
				
		input_file_raw = input("Input data : ")

		# Processes input_file to ensure it is a valid file path
		if platform.system() == "Windows":
			input_file = input_file_raw.replace("\\", "/")
		else:
			input_file = input_file_raw.replace("\\", "")
		input_file = input_file.replace("\"", "")
		
		if input_file.startswith("~"):
			input_file = os.path.expanduser(input_file)
			
		# Validates input_file 
		if input_file == "":
			print("\n! Invalid input. Please input the name of a file." + help_text)
			time.sleep(2)
		elif input_file == "quit":
			save_settings(input_file, percentile, axis_max, photo_orientation)
			print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
			sys.exit()
		elif input_file == "help":
			print(help_text)
			input("Press any key to continue.")
		elif os.path.exists(input_file) == False:
			if os.path.exists(input_file + ".csv") == True:
				input_file = input_file + ".csv"
				wait_for_file = True
			elif os.path.exists(input_file[:-1]) == True:
				input_file = input_file[:-1]
				if input_file.endswith(".csv") == True:
					wait_for_file = True
				else:	
					print("\n! Invalid input. Please provide a CSV file." + help_text)
					time.sleep(2)
			else:
				print("\n! File not found. Please try again." + help_text)
				time.sleep(2)
		else:
			if input_file.endswith(".csv") == True:
				wait_for_file = True
			else:	
				print("\n! Invalid input. Please provide a CSV file." + help_text)
				time.sleep(2)

	proceed_lvl_one = False
	while proceed_lvl_one == False: 

		proceed_lvl_two = False
		while proceed_lvl_two == False: # Input UI loop
		
			clear()
			print("STRUCTURAL ANISOTROPY CALCULATOR v1\n***********************************\n")

			print("Input data : " + input_file)
		
			print("\n\tSETTINGS\n\t")
			print("\t1. Percentile = " + str(percentile))
			print("\t2. Orientation = " + str(photo_orientation))
			print("\t3. Axis max = " + str(axis_max) + "\n")

			control_code = str(input("To change settings, type setting number. To proceed, hit return  : "))
		
			if control_code == "":
				print("\nCalculating…")
				proceed_lvl_two= True
			elif control_code == "1":				
				wait_for_input = False
				while wait_for_input == False:
					percentile = input("\nSpecify percentile : ")
					
					if percentile == "":
						print("\n! Invalid input. Please enter a value for the percentile." + help_text)
						time.sleep(2)
					elif percentile == "quit":
						save_settings(input_file, percentile, axis_max, photo_orientation)
						print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
						sys.exit()
					elif percentile.isdigit() == False:
						print("\n! Invalid input. Please input a numerical value." + help_text)
						time.sleep(2)
					else:
						percentile = int(percentile)
						wait_for_input = True
				
			elif control_code == "2":				
				wait_for_input = False
				while wait_for_input == False:
					photo_orientation = input("\nSpecify the angle to north in input image : ")
					
					if photo_orientation == "none":
						wait_for_input = True
					elif photo_orientation == "quit":
						save_settings(input_file, percentile, axis_max, photo_orientation)
						print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
						sys.exit()
					elif photo_orientation == "":
						print("\n! Invalid input. Please enter a value for the orientation." + help_text)
						time.sleep(2)
					elif photo_orientation.isdigit() == False:
						print("\n! Invalid input. Please input a numerical value." + help_text)
						time.sleep(2)
					else:
						wait_for_input = True
				
			elif control_code == "3":
				wait_for_input = False
				while wait_for_input == False:
					axis_max = input("\nSpecify maximum aspect ratio to display : ")
					
					if axis_max == "quit":
						save_settings(input_file, percentile, axis_max, photo_orientation)
						print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
						sys.exit()
					elif axis_max == "":
						print("\n! Invalid input. Please enter a maximum axis value." + help_text)
						time.sleep(2)
					elif axis_max.isdigit() == False:
						print("\n! Invalid input. Please input a numerical value." + help_text)
						time.sleep(2)
					else:
						axis_max = int(axis_max)
						wait_for_input = True
			elif control_code == "quit":
				save_settings(input_file, percentile, axis_max, photo_orientation)
				print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
				sys.exit()
			elif control_code == "help":
				print(help_text)
				input("Press any key to continue.")
			else:
				print("\n! Invalid input. Please select again." + help_text)
				time.sleep(2)


			processed_data = process_data(input_file, percentile, axis_max, photo_orientation)

			detrended_data = processed_data[0]
			detrended_xy_data = processed_data[1]
			retrended_data = processed_data[2]
			retrended_xy_data = processed_data[3]
			rounded_structural_anisotropy = processed_data[4]
			retrend_by = processed_data[5]
			trend = processed_data[6]
			ellipse_axes = processed_data[7]


		proceed_lvl_two= False
		while proceed_lvl_two == False: # Output UI loop
	
			clear()
			print("STRUCTURAL ANISOTROPY CALCULATOR v1\n***********************************\n")
		
			print("Input data : " + input_file)

			SA_value = "SA" + str(percentile) + " = " + str(rounded_structural_anisotropy)
			print("\nRESULTS:  " + SA_value + "\t(X = " + str(numpty.round(ellipse_axes[1], 2)) + ", Y = " + str(numpty.round(ellipse_axes[0], 2)) + ")")

			if str(photo_orientation) != "none":
				trend_value = print("\t  Trend = " + str(trend) + "\n")
	
			print("\n\tOPTIONS\n")
			print("\t1. Preview plot")
			print("\t2. Save plot")
			print("\t3. Export data")
			print("\t4. Change settings")
			print("\t5. New calculation")
			print("\t6. Quit\n")

			control_code = str(input("Select an option  : "))

			if control_code == "1":
				draw_plot("preview", "")
			elif control_code == "2":
			
				proceed_lvl_three = False
				while proceed_lvl_three == False: 
					print("\n\tSAVE PLOT\n")
					print("\t1. Save plot as png")
					print("\t2. Save plot as pdf")
					print("\t3. Save plot as svg")
					print("\t4. Back to options menu\n")
					
					control_code_lvl_three = str(input("Select format to save as  : "))
					
					
					if control_code_lvl_three == "1":
						file_name = str(input("Save as  : "))
						draw_plot("save_raster", file_name)
						print("\nSaving…")
						time.sleep(1)
						proceed_lvl_three = True
					elif control_code_lvl_three == "2":
						file_name = str(input("Save as  : "))
						draw_plot("save_vector", file_name)
						print("\nSaving…")
						time.sleep(1)
						proceed_lvl_three = True
					elif control_code_lvl_three == "3":
						file_name = str(input("Save as  : "))
						draw_plot("save_vector_svg", file_name)
						print("\nSaving…")
						time.sleep(1)
						proceed_lvl_three = True
					elif control_code_lvl_three == "4":
						proceed_lvl_three = True
					elif control_code_lvl_three == "quit":
						save_settings(input_file, percentile, axis_max, photo_orientation)
						print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
						sys.exit()
					elif control_code_lvl_three == "help":
						print(help_text)
						input("Press any key to continue.")
					else:
						print("\n! Invalid input. Please select again." + help_text)
						time.sleep(2)		
			elif control_code == "3":
		
				proceed_lvl_three = False
				while proceed_lvl_three == False: 
					print("\n\tEXPORT DATA\n")
					print("\t1. Output data")
					print("\t2. Detrended output data")
					print("\t3. XY data")
					print("\t4. Detrended XY data")
					print("\t5. Back to options menu\n")
			
					control_code_lvl_three = str(input("Select data to export  : "))
			
					if control_code_lvl_three == "1":
						file_name = str(input("Save as  : "))
						numpty.savetxt(file_name + ".csv", retrended_data, delimiter=",")
						print("\nSaving…")
						time.sleep(1)
						proceed_lvl_three = True
					elif control_code_lvl_three == "2":
						file_name = str(input("Save as  : "))
						numpty.savetxt(file_name + ".csv", detrended_data, delimiter=",")
						print("\nSaving…")
						time.sleep(1)
						proceed_lvl_three = True
					elif control_code_lvl_three == "3":
						file_name = str(input("Save as  : "))
						numpty.savetxt(file_name + ".csv", retrended_xy_data, delimiter=",")
						print("\nSaving…")
						time.sleep(1)
						proceed_lvl_three = True
					elif control_code_lvl_three == "4":
						file_name = str(input("Save as  : "))
						numpty.savetxt(file_name + ".csv", detrended_xy_data, delimiter=",")
						print("\nSaving…")
						time.sleep(1)
						proceed_lvl_three = True
					elif control_code_lvl_three == "5":
						proceed_lvl_three = True
					elif control_code_lvl_three == "quit":
						save_settings(input_file, percentile, axis_max, photo_orientation)
						print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
						sys.exit()
					elif control_code_lvl_three == "help":
						print(help_text)
						input("Press any key to continue.")
					else:
						print("\n! Invalid input. Please select again." + help_text)
						time.sleep(2)

	
				print("\nSave plot\n")
			elif control_code == "4":
				proceed_lvl_two = True
			elif control_code == "5":
				proceed_lvl_two = True
				proceed_lvl_one = True
			elif control_code == "6":
				save_settings(input_file, percentile, axis_max, photo_orientation)
				print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
				proceed_lvl_two = True
				proceed_lvl_one = True
				proceed_lvl_zero = True
			elif control_code == "quit":
				save_settings(input_file, percentile, axis_max, photo_orientation)
				print("\nQuitting… Thank you for using Structural Anisotropy Calculator.\n")
				sys.exit()
			elif control_code == "help":
				print(help_text)
				input("Press any key to continue.")
			else:
				print("\n! Invalid input. Please select again." + help_text)
				time.sleep(2)
