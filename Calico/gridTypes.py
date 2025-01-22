import numpy as np

class HexGrid:
    def __init__(self, data):
        # information is stored in a dictionary
        # get_item has been implemented to allow indexing across all rows
        if type(data) == list:
            data = HexGrid.basicToCustom(data)
        self.data = data
        self.rowdata = HexGrid.calculateRowdata(data)
        print(self.data)
    
    def basicToCustom(data):
        # code to turn a list of lists into a hex grid. Assumes first thing along each
        # line goes in i=0, k+=1
        d2 = {}
        i = 0
        j = 0
        k = 0
        for sublist in data:
            i = -k
            j = k
            for value in sublist:
                if value is not None:
                    d2[i,j] = value
                i+=1
            k+=1
        return d2
    
    def calculateRowdata(hexdata):
        # expect the hexdata to be a basic hexgrid dictionary.
        # this takes that data and extracts the max and min i,j values for each 
        # single dimension (i = 0, k = 3 etc)
        rowdata={}
        k = np.array(list(hexdata.keys()))
        keys = np.zeros((len(k),3),dtype=int)
        keys[:,0:2] = k
        keys[:,2] = k[:,0] + k[:,1]
        i_inds = np.unique(keys[:,0])
        j_inds = np.unique(keys[:,1])
        k_inds = np.unique(keys[:,2])
        for i in i_inds:
            limited_list = keys[keys[:,0] == i]
            rowdata[0,i] = [i,i+1,np.min(limited_list,axis=0)[1],np.max(limited_list, axis=0)[1]+1]
        for j in j_inds:
            limited_list = keys[keys[:,1] == j]
            rowdata[1,j] = [np.min(limited_list,axis=0)[0],np.max(limited_list, axis=0)[0]+1, j,j+1]
        for k in k_inds:
            limited_list = keys[keys[:,2] == k]
            limited_list = keys[keys[:,2] == k]
            rowdata[2,k] = [np.min(limited_list,axis=0)[0],np.max(limited_list, axis=0)[0]+1,
                            np.min(limited_list,axis=0)[1],np.max(limited_list, axis=0)[1]+1]
        return rowdata
    
    def extractSlice(sliceObject,rowdata):
        start, stop, step = sliceObject.start, sliceObject.stop, sliceObject.step
        istart = rowdata[0] if start is None else start
        istop = rowdata[1] if stop is None else stop
        step = 1 if step is None else step
        jstart = rowdata[2] if start is None else start
        jstop = rowdata[3] if stop is None else stop
        iIndicies = list(range(istop-1, istart-1, -step))
        jIndicies = list(range(jstart, jstop, step))

        if (len(iIndicies) == 1) | (len(jIndicies) == 1):
            indices = [[i,j] for j in jIndicies for i in iIndicies]
        else:
            indices = [[i,j] for i,j in zip(iIndicies,jIndicies)]
        return indices
    
    def __getitem__(self, indicies):
        if not isinstance(indicies, tuple):
            indicies = tuple((indicies,slice(None),0))
        if len(indicies) == 1:
            indicies = tuple((indicies,slice(None),0))
        elif len(indicies) == 2:
            indicies = tuple((indicies[0], indicies[1], 0))
        elif len(indicies) > 3:
            raise IndexError('Too many indicies')
        points = []
        if type(indicies[0]) == slice:
            points = np.array(HexGrid.extractSlice(indicies[0],self.rowdata[1,indicies[1]]))
            points = points[::-1]
        elif type(indicies[0]) == int:
            i = indicies[0]
        elif type(indicies[0]) == str:
            points = np.array(HexGrid.extractSlice(slice(None),self.rowdata[2,indicies[1]]))
        else:
            i = 0
        if type(indicies[1]) == slice:
            points = np.array(HexGrid.extractSlice(indicies[1],self.rowdata[0,indicies[0]]))
        elif type(indicies[1]) == int:
            j = indicies[1]
        else:
            j = 0
        if type(indicies[2]) == slice:
            raise IndexError('K dimension should be None or int, not slice. set first dim to "k" to get rows in k dim')
        elif type(indicies[2]) == int:
            k = indicies[2]
        else:
            k = 0
        if len(points) == 0:
            i = i - k
            j = j + k
            if (type(i) == int) and (type(j) == int):
                return self.data[i,j]
        else:
            return [self.data[a[0],a[1]] for a in points]
            
        
        

a = HexGrid([[1],[2,3],[None,4]])

print('i rows')
print('i: -1 :',a[-1,:])
print('i: 0 :',a[0,:])
print('J rows')
print('j: 0 :',a[:,0])
print('j: 1 :',a[:,1])
print('j: 2 :',a[:,2])
print('K rows')
print('k: 0 :',a['k',0])
print('k: 1 :',a['k',1])