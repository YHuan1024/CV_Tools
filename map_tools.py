import xml.etree.ElementTree as ET
import os
import pickle as cPickle 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
class VocMap:
	'''
	model_output_path: Path to detections
        detpath.format(classname) should produce the detection results file.
    annotations_path: Path to annotations
        annopath.format(imagename) should be the xml annotations file. #xml 标注文件。
    imageset_file: Text file containing the list of images, one image per line. 
    cachedir: Directory for caching the annotations #缓存标注的目录路径VOCdevkit/annotation_cache,图像数据只读文件，为了避免每次都要重新读数据集原始数据。
    [ovthresh]: Overlap threshold (default = 0.5) #iou
    [use_07_metric]: Whether to use VOC07's 11 point AP computation 
        (default False) #是否使用VOC07的AP计算方法，voc07是11个点采样。
	'''
	def __init__(self, model_output_path, annotations_path, imageset_file, cache_dir='.'):
		self.model_output_path = model_output_path
		self.annotations_path = annotations_path
		self.imageset_file = imageset_file
		self.cache_dir = cache_dir

	def parse_rec(self, filename): #读取标注的xml文件
	    """ Parse a PASCAL VOC xml file """
	    tree = ET.parse(filename)
	    objects = []
	    
		    for obj in tree.findall('object'):
		        obj_struct = {}
		        obj_struct['name'] = obj.find('name').text
		        obj_struct['pose'] = obj.find('pose').text
		        obj_struct['truncated'] = int(obj.find('truncated').text)
		        obj_struct['difficult'] = int(obj.find('difficult').text)
		        bbox = obj.find('bndbox')
		        obj_struct['bbox'] = [int(bbox.find('xmin').text),
		                              int(bbox.find('ymin').text),
		                              int(bbox.find('xmax').text),
		                              int(bbox.find('ymax').text)]
		        objects.append(obj_struct)
		
	    return objects

	def voc_ap(self, rec, prec, use_07_metric=False):
	    """ ap = voc_ap(rec, prec, [use_07_metric])
	    Compute VOC AP given precision and recall.
	    If use_07_metric is true, uses the
	    VOC 07 11 point method (default:False).
	    计算AP值，若use_07_metric=true,则用11个点采样的方法，将rec从0-1分成11个点，这些点prec值求平均近似表示AP
	    若use_07_metric=false,则采用更为精确的逐点积分方法
	    """
	    if use_07_metric:
	        # 11 point metric
	        ap = 0.
	        for t in np.arange(0., 1.1, 0.1):
	            if np.sum(rec >= t) == 0:
	                p = 0
	            else:
	                p = np.max(prec[rec >= t])
	            ap = ap + p / 11.
	    else:
	        # correct AP calculation
	        # first append sentinel values at the end
	        mrec = np.concatenate(([0.], rec, [1.]))# 拼接
	        mpre = np.concatenate(([0.], prec, [0.]))
	 
	        # compute the precision envelope
	        for i in range(mpre.size - 1, 0, -1):
	            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
	 
	        # to calculate area under PR curve, look for points
	        # where X axis (recall) changes value
	        i = np.where(mrec[1:] != mrec[:-1])[0]
	 
	        # and sum (\Delta recall) * prec
	        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
	    return ap
	def voc_eval(self, class_name, ovthresh=0.5,use_07_metric=False):

	    # first load gt 加载ground truth。
	    if not os.path.isdir(self.cache_dir):
	        os.mkdir(self.cachedir)
	    cachefile = os.path.join(self.cache_dir, 'annots.pkl') #只读文件名称。
	    # read list of images
	    with open(self.imageset_file, 'r') as f:
	        lines = f.readlines() #读取所有待检测图片名。
	    imagenames = [x.strip() for x in lines] #待检测图像文件名字存于数组imagenames,长度1000。
	 
	    if not os.path.isfile(cachefile): #如果只读文件不存在，则只好从原始数据集中重新加载数据
	        # load annots
	        recs = {}
	        for i, imagename in enumerate(imagenames):
	            recs[imagename] = self.parse_rec(self.annotations_path.format(imagename)) #parse_rec函数读取当前图像标注文件，返回当前图像标注，存于recs字典（key是图像名，values是gt）
	            if i % 100 == 0:
	                print('Reading annotation for {:d}/{:d}'.format(
	                    i + 1, len(imagenames))) #进度条。
	        # save
	        print('Saving cached annotations to {:s}'.format(cachefile))
	        with open(cachefile, 'wb') as f:
	            cPickle.dump(recs, f) #recs字典c保存到只读文件。
	    else:
	        # load
	        with open(cachefile, 'rb') as f:
	            recs = cPickle.load(f) #如果已经有了只读文件，加载到recs。
	 
	    # extract gt objects for this class #按类别获取标注文件，recall和precision都是针对不同类别而言的，AP也是对各个类别分别算的。
	    class_recs = {} #当前类别的标注
	    npos = 0 #npos标记的目标数量
	    for imagename in imagenames:
	        R = [obj for obj in recs[imagename] if obj['name'] == class_name] #过滤，只保留recs中指定类别的项，存为R。
	        bbox = np.array([x['bbox'] for x in R]) #抽取bbox
	        difficult = np.array([x['difficult'] for x in R]).astype(np.bool) #如果数据集没有difficult,所有项都是0.
	 
	        det = [False] * len(R) #len(R)就是当前类别的gt目标个数，det表示是否检测到，初始化为false。
	        npos = npos + sum(~difficult) #自增，非difficult样本数量，如果数据集没有difficult，npos数量就是gt数量。
	        class_recs[imagename] = {'bbox': bbox,  
	                                 'difficult': difficult,
	                                 'det': det}
	 
	    # read dets 读取检测结果
	    detfile = self.model_output_path.format(class_name)
	    with open(detfile, 'r') as f:
	        lines = f.readlines()
	 
	    splitlines = [x.strip().split(' ') for x in lines] #假设检测结果有20000个，则splitlines长度20000
	    image_ids = [x[0] for x in splitlines] #检测结果中的图像名，image_ids长度20000，但实际图像只有1000张，因为一张图像上可以有多个目标检测结果
	    confidence = np.array([float(x[1]) for x in splitlines]) #检测结果置信度
	    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) #变为浮点型的bbox。
	 
	    # sort by confidence 将20000各检测结果按置信度排序
	    sorted_ind = np.argsort(-confidence) #对confidence的index根据值大小进行降序排列。
	    sorted_scores = np.sort(-confidence) #降序排列。
	    BB = BB[sorted_ind, :] #重排bbox，由大概率到小概率。
	    image_ids = [image_ids[x] for x in sorted_ind] #对image_ids相应地进行重排。
	 
	    # go down dets and mark TPs and FPs 
	    nd = len(image_ids) #注意这里是20000，不是1000
	    tp = np.zeros(nd) # true positive，长度20000
	    fp = np.zeros(nd) # false positive，长度20000
	    for d in range(nd): #遍历所有检测结果，因为已经排序，所以这里是从置信度最高到最低遍历
	        R = class_recs[image_ids[d]] #当前检测结果所在图像的所有同类别gt
	        bb = BB[d, :].astype(float) #当前检测结果bbox坐标
	        ovmax = -np.inf
	        BBGT = R['bbox'].astype(float) #当前检测结果所在图像的所有同类别gt的bbox坐标
	 
	        if BBGT.size > 0: 
	            # compute overlaps 计算当前检测结果，与该检测结果所在图像的标注重合率，一对多用到python的broadcast机制
	            # intersection
	            ixmin = np.maximum(BBGT[:, 0], bb[0])
	            iymin = np.maximum(BBGT[:, 1], bb[1])
	            ixmax = np.minimum(BBGT[:, 2], bb[2])
	            iymax = np.minimum(BBGT[:, 3], bb[3])
	            iw = np.maximum(ixmax - ixmin + 1., 0.)
	            ih = np.maximum(iymax - iymin + 1., 0.)
	            inters = iw * ih
	 
	            # union
	            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
	                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
	                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
	 
	            overlaps = inters / uni
	            ovmax = np.max(overlaps)#最大重合率
	            jmax = np.argmax(overlaps)#最大重合率对应的gt
	 
	        if ovmax > ovthresh:#如果当前检测结果与真实标注最大重合率满足阈值
	            if not R['difficult'][jmax]:
	                if not R['det'][jmax]:
	                    tp[d] = 1. #正检数目+1
	                    R['det'][jmax] = 1 #该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
	                else: #相反，认为检测到一个虚警
	                    fp[d] = 1.
	        else: #不满足阈值，肯定是虚警
	            fp[d] = 1.
	 
	    # compute precision recall
	    fp = np.cumsum(fp) #积分图，在当前节点前的虚警数量，fp长度
	    tp = np.cumsum(tp) #积分图，在当前节点前的正检数量

	    rec = tp / float(npos) #召回率，长度20000，从0到1
	    # avoid divide by zero in case the first detection matches a difficult
	    # ground truth 准确率，长度20000，长度20000，从1到0
	    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
	    ap = self.voc_ap(rec, prec, use_07_metric)
	 
	    return rec, prec, ap
	def compute_map(self, find_path):
		FileNames = os.listdir(find_path)
		map_voc = 0
		class_num = 0
		for file in FileNames:
			file_name = file.split('.')[0]
			rec,prec,ap = self.voc_eval(file_name)
			if not np.isnan(ap):
				map_voc += ap
				class_num += 1
		map_voc = map_voc/class_num
		#map_voc = map_voc/len(FileNames)    #map的计算方式，分母是否跟随改变
		return map_voc

	def pr_draw(self, find_path, pr_name='./yolov3_pr.png'):

		FileNames = os.listdir(find_path)
		plt.figure()
		plt.xlabel('recall')
		plt.ylabel('precision')
		plt.title('P-R curve')
		for file in FileNames:
			file_name = file.split('.')[0]
			rec,prec,ap = self.voc_eval(file_name)
			plt.plot(rec,prec, label=str(file_name))
		plt.legend()#显示图例，如果注释改行，即使设置了图例仍然不显示
		plt.savefig(pr_name)
		plt.show()

	def ap_draw(self, find_path, ap_name='./yolov3_ap.png'):
		FileNames = os.listdir(find_path)
		matplotlib.rcParams['font.sans-serif'] = ['SimHei']#设置字体
		matplotlib.rcParams['axes.unicode_minus'] = False #编码 显示中文
		aps = list()    #ap值的列表
		yticks = list() #ap值的分类列表
		plt.figure(figsize=(12, 5))  #设置图片尺寸显示完全标签
		for file in FileNames:
			file_name = file.split('.')[0]
			_,_,ap = self.voc_eval(file_name)
			if np.isnan(ap):
				ap = 0.0
			aps.append(float(format(ap, '.2f'))) #保留两位小数点
			yticks.append(file_name)
		aps = np.array(aps)
		plt.barh(range(len(FileNames)), aps, height=0.7, color='steelblue', alpha=0.8)  #水平柱状图配置
		plt.yticks(range(len(FileNames)),yticks)
		plt.xlim(0,1)
		plt.xlabel("AP")
		plt.ylabel("分类")
		plt.title("各类目标检测的ap")
		for x, y in enumerate(aps):
			plt.text(y + 0.02, x - 0.1, '%s' % y)   #柱状图末端数值
		plt.savefig(ap_name)
		plt.show()

if __name__ == '__main__':
	aa = VocMap('./results/{}.txt', './VOC2007/Annotations/{}.xml', './VOC2007/ImageSets/Main/test.txt')
	maps = aa.compute_map('./results/')
	_, _, ap = aa.voc_eval('person')
	print("ap:", ap)
	print("map:", maps)
	aa.pr_draw('./results/')
	aa.ap_draw('./results/')