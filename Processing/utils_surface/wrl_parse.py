import re

import numpy as np
from numpy import matrix, savetxt


def list2csv(myList, fname, FMT='%.6f'):
    '''
    *************************************************************************
    *myList*: a 2D list of floating point numbers                           *
    *fname*: name of the output csv file                                    *
    *************************************************************************
    *This function converts a 2D list to a csv (with index and no header)   *
    *************************************************************************
    '''
    mat = matrix(myList)  # list to matrix
    savetxt(fname, mat.astype(float), fmt=FMT, delimiter=',')


def vrml2csv(filename, root):
    with open(filename) as f:
        nodeCount = 0
        while 1:  # skip initial parameters
            ln = f.readline().split()  # python3
            # termination condition:
            eof = 0
            while ln == []:
                ln = f.readline().split()
                eof += 1
                if eof > 10:
                    return

            if (ln != []) and (ln[0] == 'point'):
                nodeCount += 1
                coord = []
                print('Reading vertex coordinates.')
                ln[4] = ln[4][:-1]
                coord.append(ln[2:5])  # first coordinate
                while 1:
                    ln = f.readline().split()
                    if len(ln) > 2:
                        ln[2] = ln[2][:-1]  # remove comma
                        coord.append(ln[0:3])
                    if ln == ['}']:
                        break

                # get normal
                print('Reading normal vectors.')
                normalVector = []
                f.readline()  # normal
                f.readline()  # Normal {
                ln = f.readline().split()
                ln[4] = ln[4][:-1]  # remove comma
                normalVector.append(ln[2:5])
                while 1:
                    ln = f.readline().split()
                    if len(ln) > 2:
                        if ln[2].endswith(','):
                            ln[2] = ln[2][:-1]
                        normalVector.append(ln[0:3])
                    if ln == ['}']:
                        list2csv(normalVector, root + '_normal_' + str(nodeCount) + '.csv')
                        break
                # then get coordIndex
                print('Reading coordinate indices.')
                coordIndex = []
                ln = f.readline().split()  # first coordIndex
                coordIndex.append([ln[2][:-1], ln[3][:-1], ln[4][:-1]])
                coordIndex.append([ln[6][:-1], ln[7][:-1], ln[8][:-1]])
                while 1:
                    ln = f.readline().split()
                    if len(ln) > 7:
                        coordIndex.append([ln[0][:-1], ln[1][:-1], ln[2][:-1]])
                        coordIndex.append([ln[4][:-1], ln[5][:-1], ln[6][:-1]])
                    if len(ln) == 9:
                        list2csv(coordIndex, root + '_index_' + str(nodeCount) + '.csv',
                                 FMT='%.0f')
                        break


def read_wrl2(filename, verbose=0):
    # read_wrl - load a mesh from a VRML file
    with open(filename, 'r') as fp:

        tempstr = ' '
        nodeCount = 0
        coord = []
        face = []
        color = []
        normal = []
        ln = ['a']
        endfile = False

        while not endfile:
            line = fp.readline()
            if len(line) == 0:
                endfile = True
                break
            ln = line.split()
            # point coordinates
            if (ln != []) and (ln[0] == 'point'):

                if verbose: print('Reading vertex coordinates.')
                end_section = False
                while not end_section:
                    lns = fp.readline().split(',')
                    for tris in lns:
                        ln = tris.split()
                        if len(ln) == 0:
                            continue
                        if ln[0] == ']':
                            # coord = matrix(coord).astype(float)
                            end_section = True
                            break
                        if len(ln) > 2:
                            # ln[2] = ln[2][:-1]  # remove comma
                            coord.append(np.asarray(ln, dtype=float))

            # faces
            if (ln != []) and (ln[0] == 'coordIndex'):
                if verbose: print('Reading faces indexes.')
                while 1:
                    line = fp.readline()
                    ln = list(filter(None, re.split(' |,|\n', line)))
                    if ln == [']']:
                        # face = matrix(face)
                        break
                    if len(ln) > 2:
                        for i_ln in range(0, len(ln), 4):
                            face.append(np.asarray(ln[i_ln:i_ln + 3], dtype=int))
            # color
            if (ln != []) and (ln[0] == 'color'):

                if verbose: print('Reading color vertex.')
                end_section = False
                while not end_section:
                    lns = fp.readline().split(',')
                    for tris in lns:
                        ln = tris.split()
                        if len(ln) == 0:
                            continue
                        if ln[0] == ']':
                            # coord = matrix(coord).astype(float)
                            end_section = True
                            break
                        if len(ln) > 2:
                            # ln[2] = ln[2][:-1]  # remove comma
                            color.append(np.asarray(ln, dtype=float))

            # normal
            if (ln != []) and (ln[0] == 'vector') and ln[1] == '[':
                if verbose: print('Reading normal vertex.')
                while 1:
                    ln = fp.readline().split()
                    if ln[0] == ']':
                        # normal = matrix(normal)
                        endfile = True
                        break
                    if len(ln) > 2:
                        ln[-1] = ln[-1][:-1]
                        normal.append(np.asarray(ln[0:3], dtype=float))
    return np.array(coord), np.array(face), np.array(color), np.array(normal)



if __name__ == "__main__":
    read_wrl2(r"..\data_epipred\data_train\wrl\1ahw.wrl")