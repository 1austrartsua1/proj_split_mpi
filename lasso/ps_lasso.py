




def create_simple_partition(n,nparts):
    #
    partsize = int(n/nparts)
    partition = [range(i*partsize,(i+1)*partsize) for i in range(nparts)]
    if n%nparts != 0:
        partition.append(range(nparts*partsize,n))

    return partition
