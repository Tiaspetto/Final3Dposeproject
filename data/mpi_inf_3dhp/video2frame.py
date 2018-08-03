import os

for ind in range(5,9):
	for seq in range(1,3):
		for v in range(9):

			print("please input loop flag, S_INDEX, sequence_index, video_index")
			command = 'ffmpeg -i "G:/mpi_inf_3dhp/S{index}/Seq{sequence}/imageSequence/video_{video}.avi" -qscale:v 1 "G:/mpi_inf_3dhp/S{index}/Seq{sequence}/imageSequence/frames/v{video}/img{video}_%06d.jpg"'
			command = command.format(index = ind, sequence = seq, video = v)
			print("extract frames to path： "+command)

			os.system(command)

