# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

from Utilities import saveDict

class FeatPoint:
	"""One feature point, top-left corner of the checker
	"""
	def __init__(self, row, col, **kwargs):
		"""
		Args:
			row (str or int): Label of the row. 'A' = 1, 'Z' = 26
			col (str or int): Label of the column.
			kwargs (dict):
				'rawPoints' (list): List of rawPoints, for loading data

		Note:
			rawPoints = [[point, view], ...]
			# feature point coordinate on sensor plane and its corresponding view
			# turn into 3D coordinates
			# on sensor plane, z = 0
		"""
		self.row = row.upper() if type(row) == str else chr(64+row)
		self.col = col.upper() if type(col) == str else chr(64+col)
		self.rawPoints = kwargs['rawPoints'] if 'rawPoints' in kwargs else []

	def __repr__(self):
		return 'Feature point ' + self.row + self.col

	def __len__(self):
		return len(self.rawPoints)

	def add(self, point, view):
		"""Add a point from a view.
		Args:
			point (tuple): Raw global coordinate (on sensor plane) of a feature point. (y, x)
			view (tuple): The view where the point is. (row, col), NOT the same as row/col in __init__
		"""
		self.rawPoints.append([np.array([point[0], point[1], 0], dtype='float64'), view])

	def delete(self, point):
		"""Delete a point from a view
		Args:
			point (tuple): A point close to the one to be deleted.
		"""
		dmin, index = np.inf, -1
		for ii, content in enumerate(self.rawPoints):
			p, _ = content
			d = (p[0]-point[0])**2 + (p[1]-point[1])**2
			if d < dmin:
				dmin = d
				index = ii
		pv = self.rawPoints.pop(index)
		print('point', pv[0],'in view', pv[1], 'deleted')

	@property
	def isEmpty(self):
		return len(self.rawPoints) == 0

	@property
	def isValid(self):
		# at least 3 points to estimate min circle
		return len(self.rawPoints) > 2

	def save(self, filename):
		"""Save self.row, self.col, self.rawPoints
		Args:
			filename (str): filename, json file
		"""
		data = {'row':self.row, 'col':self.col}
		data['rawPoints'] = []
		for rawP, view in self.rawPoints:
			p_list = list(rawP)
			data['rawPoints'].append([p_list, view])

		saveDict(filename, data)

class RayTraceModule:
	"""First order ray tracing (MLA)
	"""
	def __init__(self, M_MLA, f_MLA, centers, fp, **kwargs):
		"""
		Args:
			M_MLA (float): Magnification of MLA. (=z_obj/z_img)
			f_MLA (float): Focal length of MLA, [pixel]
			centers (numpy.ndarray): Centers of each view. Should be a (#row x #col x 2) array.
			fp (class FeatPoint): Feature point.
			kwargs:
				row_center (int): Index of center row
				col_center (int): Index of center col
				II_norm_vec (numpy.ndarray): Normal vector of II plane.
		"""
		self.M_MLA = M_MLA
		self.f_MLA = f_MLA
		self.centers = centers
		self.fp = fp
		self._update_first_order(self.M_MLA)

		self.row_center = kwargs.get('row_center', 0)
		self.col_center = kwargs.get('col_center', 0)
		self.row_total, self.col_total, _ = self.centers.shape
		self.centers_3d = self._update_center_3d(self.centers)

		self.II_norm_vec = kwargs.get('II_norm_vec', np.array([0,0,1],dtype='float64'))
		self.II_norm_vec /= np.linalg.norm(self.II_norm_vec)

	def __repr__(self):
		return 'Feature point ' + self.fp.row + self.fp.col + ' -- RayTraceModule'

	def _update_first_order(self, M):
		self.z_obj = (M+1)*self.f_MLA
		self.z_img = self.z_obj/M

	def _update_center_3d(self, centers):
		centers_3d = np.zeros((centers.shape[0], centers.shape[1], 3), dtype='float64')
		centers_3d[:, :, :2] = centers
		centers_3d[:, :, -1] = self.z_img
		return centers_3d

	def _isCollinear(self, v1, v2):
		"""
		Determines whether two vectors are collinear.
		Returns True if the vectors are collinear, False otherwise.
		© ChatGPT
		"""
		# Calculate the cross product between the vectors
		cross_product = np.cross(v1, v2)
		# Check if the magnitude of the cross product is zero (within a tolerance)
		return np.isclose(np.linalg.norm(cross_product), 0.0, rtol=1e-10)

	def project(self, **kwargs):
		"""Project feature points to II plane
		Args:
			kwargs:
				1. update self variables
					points (list): Points to project, [[p(1x3), view(1x2)], ...]
					M_MLA (float): Custom M_MLA.
					II_norm_vec (numpy.ndarray): II plane norm vector, (1x3)
					centers (numpy.ndarray): Projection centers, (#row x #col x 2)
				2. new variables
					dist (list): Distortion coefficients, [k1, k2, k3]
					norm (float): Normalization factor.
		"""
		M = kwargs.get('M_MLA', self.M_MLA)
		self._update_first_order(M)
		points_to_project = kwargs.get('points', self.fp.rawPoints)
		if 'II_norm_vec' in kwargs:
			self.II_norm_vec = kwargs['II_norm_vec'] / np.linalg.norm(kwargs['II_norm_vec'])
		if 'centers' in kwargs:
			centers = kwargs['centers']
			centers_3d = self._update_center_3d(centers)
		else:
			centers = self.centers
			centers_3d = self.centers_3d

		if len(points_to_project) == 0:
			raise ValueError('No point to be projected')

		# undistort
		if 'dist' in kwargs:
			points_undist = []
			dist = kwargs['dist']
			norm = kwargs['norm']

			for rawP, view in points_to_project:
				center = centers_3d[view[0], view[1], :]
				local_p_vec = rawP - center

				local_norm_p_vec = local_p_vec / norm
				local_norm_r_2 = np.sum(local_norm_p_vec**2)
				factor = 1
				for i in range(len(dist)):
					factor += dist[i]*local_norm_r_2**(i+1)
				local_norm_p_vec_dist = local_norm_p_vec * factor

				p_vec_dist = local_norm_p_vec_dist * norm + center
				points_undist.append([p_vec_dist, view])

			points_to_project = points_undist

		# project to II plane
		IIpoints = []
		xp, yp, zp = centers[self.row_center, self.col_center, 1], centers[self.row_center, self.col_center, 0], self.z_obj+self.z_img
		ap, bp, cp = self.II_norm_vec # represent a plane: a(x-x0)+b(y-y0)+c(z-z0)=0
		for p, view in points_to_project:
			center = centers_3d[view[0], view[1], :]
			dir_vec = center - p # direction vector (a, b, c); represent a line: (x-x0)/a = (y-y0)/b = (z-z0)/c
			al, bl, cl = dir_vec
			# solve intersection of line & plane
			A = np.array([[ap, bp, cp], [1/al, -1/bl, 0], [0, 1/bl, -1/cl]])
			B = np.array([[ap*xp+bp*yp+cp*zp], [p[0]/al-p[1]/bl], [p[1]/bl-p[2]/cl]])
			xyz = np.matmul(np.linalg.inv(A), B).reshape(3)
			IIpoints.append(xyz)
		self.IIpoints = np.array(IIpoints)
		return self.IIpoints

	@property
	def minCircle(self):
		"""Error function, rotate II plane normal to z-axis, then calculate min circle
		Returns:
			minCircle (dict): Center and radius of the smallest circle enclosing all reprojected points. {'r':r,'c':(y,x)}
		"""
		v_init = self.II_norm_vec
		v_rot = np.array([0, 0, 1]) # optical axis, z-axis

		if self._isCollinear(v_init, v_rot):
			theta = 0
			rot_matrix = np.eye(3)
		else:
			k = np.cross(v_init, v_rot)
			k /= np.linalg.norm(k)

			# Calculate the rotation matrix using the Rodrigues rotation formula
			theta = np.arccos(np.dot(v_init, v_rot) / (np.linalg.norm(v_init) * np.linalg.norm(v_rot)))
			cos_theta = np.cos(theta)
			sin_theta = np.sin(theta)

			rot_matrix = np.array([
				[
				cos_theta + (1-cos_theta)*k[0]**2,
				(1-cos_theta)*k[0]*k[1] - sin_theta*k[2],
				(1-cos_theta)*k[0]*k[2] + sin_theta*k[1]
				],
				[
				(1-cos_theta)*k[0]*k[1] + sin_theta*k[2],
				cos_theta + (1-cos_theta)*k[1]**2,
				(1-cos_theta)*k[1]*k[2] - sin_theta*k[0]
				],
				[
				(1-cos_theta)*k[0]*k[2] - sin_theta*k[1],
				(1-cos_theta)*k[1]*k[2] + sin_theta*k[0],
				cos_theta + (1-cos_theta)*k[2]**2
				]
				])

			# Apply the rotation matrix to the vector
			# v_rot = np.dot(rot_matrix, v_init)

		try:
			IIpoints = self.IIpoints
		except AttributeError:
			print('Need to project all feature point to intermediate image plane first!')
			raise

		# rotate so that II plane is normal to z-axis
		origin = self.centers[self.row_center, self.col_center, :]
		origin = np.array([origin[0], origin[1], self.z_obj+self.z_img], dtype='float64')
		IIpoints -= origin
		IIpoints_rot = []
		for i in range(IIpoints.shape[0]):
			p0 = IIpoints[i, :]
			p = np.dot(rot_matrix, p0)
			IIpoints_rot.append(p)
		IIpoints_rot = np.array(IIpoints_rot)
		IIpoints_rot += origin

		points_2d = IIpoints_rot[:, :2]
		r_center, radius = cv.minEnclosingCircle(points_2d.astype('float32'))
		r_center_3d = np.array([r_center[0]-origin[0],r_center[1]-origin[1], 0])

		# back to tilted II plane
		r_center_3d_rot = np.matmul(np.linalg.inv(rot_matrix), r_center_3d)
		r_center_3d_rot += origin
		return {'center':r_center_3d_rot, 'radius':radius}
