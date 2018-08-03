import h5py
import os
import cv2
import numpy as np
import scipy.io

for ind in range(1, 2):
	for seq in range(1, 2):
		data_path = 'G:/mpi_inf_3dhp/S{index}/Seq{sequence}/annot.mat'
		data_path = data_path.format(index=ind, sequence=seq)
		# load annot
		annot = scipy.io.loadmat(data_path)
		univ_annot = annot["univ_annot3"]
		# camera set
		for v in range(2):
			annot3 = univ_annot[v][0]
			# print(np.shape(annot3))
			maxframe = np.shape(annot3)[0]
			batch_num = maxframe // 1024
			remainframe = 0
			if batch_num*1024 != maxframe:
				remainframe = maxframe - batch_num * 1024

			Path = "G:/mpi_inf_3dhp/S{index}/Seq{sequence}/imageSequence/frames/v{video}/"
			Path = Path.format(index=ind, sequence=seq, video=v)
			print(Path)
			print(batch_num)
			# process batches

			file = h5py.File(Path+"batches", 'w')
			for i in range(batch_num):
				batch_name = "{batchid}"
				batch_name = batch_name.format(batchid=i)
				start_frame = i*512

				batch_data_images = []
				batch_data_annot = []
				for fs in range(1, 513):
					current_frame = start_frame+fs
					img_path = Path+"img{video}_{frames}.jpg"
					s_current_frame = str(current_frame)
					s_current_frame = s_current_frame.zfill(6)
					img_path = img_path.format(frames=s_current_frame, video=v)

					if os.path.exists(img_path):
						srcImg = cv2.imread(img_path)
						srcImg = cv2.resize(srcImg,(512,512),interpolation=cv2.INTER_CUBIC)
						#cv2.imshow('img', srcImg)
						#cv2.waitKey()
						batch_data_images.append(srcImg)
						batch_data_annot.append(annot3[current_frame])
						print(fs)

				file.create_dataset(batch_name+"images", data=batch_data_images)
				file.create_dataset(batch_name+"annot", data=batch_data_annot)
			file.close()