#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <iostream>
#include <queue>

using namespace std;

int selfn = 0;
int selfn_ = 0;
int multi = 0;
int num_walls = 0;

static PyObject* setup(PyObject* self, PyObject* args)
{
	int size;
	if (!PyArg_ParseTuple(args, "i", &size)) {
		cout << "setup returned none!\n";
		return NULL;
	}
	selfn_ = size - 1;
	selfn = 2*size - 1;
	num_walls = 2*selfn_*selfn_;
	Py_RETURN_NONE;
}

int getItem(int* arr, int x, int y) {
	return *(arr + x*selfn + y);
}

void print_2d(int *arr) {
	for (int x = 0 ; x < selfn ; ++x) {
		for (int y = 0 ; y < selfn ; ++y) {
			cout << *(arr + x*selfn + y) << " ";
		}
		cout << endl;
	}
}

bitset<128> possible_walls(int *block) {

	bitset<128> available;
	int counter = 0;

	// h walls
	for (int x = 1 ; x < selfn ; x+=2) {
		for (int y = 0 ; y < selfn-2 ; y+=2) {
			if (!getItem(block, x, y) && !getItem(block, x, y+2) && ((!getItem(block, x-1, y+1) || !getItem(block, x+1, y+1)) || (x-3>=0 && x+3<selfn && getItem(block, x-1, y+1) && getItem(block, x-3, y+1) && getItem(block, x+1, y+1) && getItem(block, x+3, y+1))))
				available.set(counter);
			++counter;
		}
	}

	// v walls
	for (int x = 2 ; x < selfn ; x+=2) {
		for (int y = 1 ; y < selfn ; y+=2) {
			if (!getItem(block, x, y) && !getItem(block, x-2, y) && ((!getItem(block, x-1, y-1) || !getItem(block, x-1, y+1)) || (y-3>=3 && y+3<selfn && getItem(block, x-1, y-1) && getItem(block, x-1, y-3) && getItem(block, x-1, y+1) && getItem(block, x-1, y+3))))
				available.set(counter);
			++counter;
		}
	}

	return available;
}

void clean_next_visits(queue <pair<int,int> > &next_, int x, int y) {
	queue <pair<int,int> > next;
	next.push(make_pair(x,y));
	swap(next_, next);
	return;
}

void setItem3(int *arr, int x, int y, int val1, int val2) {
	*(arr + ((x*selfn + y) * 2)    ) = val1;
	*(arr + ((x*selfn + y) * 2) + 1) = val2;
}

void add_next_visits(queue <pair<int,int> > &next, int x, int y, int x_target, int *block, int *visited, int *parent) {

	*(visited + selfn*x + y) = 1;

	if (x - 2 >= 0 && !getItem(block, x-1, y) && !getItem(visited, x-2, y)){
		next.push(make_pair(x-2,y));
		setItem3(parent, x-2, y, x, y);
		*(visited + (x-2)*selfn + y) = 1;
		if (x-2 == x_target) return clean_next_visits(next, x-2, y);
	}
	if (x + 2 <= selfn-1 && !getItem(block, x+1, y) && !getItem(visited, x+2, y)){
		next.push(make_pair(x+2,y));
		setItem3(parent, x+2, y, x, y);
		*(visited + (x+2)*selfn + y) = 1;
		if (x+2 == x_target) return clean_next_visits(next, x+2, y);
	}
	if (y - 2 >= 0 && !getItem(block, x, y-1) && !getItem(visited, x, y-2)){
		next.push(make_pair(x,y-2));
		setItem3(parent, x, y-2, x, y);
		*(visited + x*selfn + y-2) = 1;
	}
	if (y + 2 <= selfn-1 && !getItem(block, x, y+1) && !getItem(visited, x, y+2)){
		next.push(make_pair(x,y+2));
		setItem3(parent, x, y+2, x, y);
		*(visited + x*selfn + y+2) = 1;
	}

}

int index_of_action(int a, int x, int y) {
	if (a == 8)
		return (x/2)*selfn_ + (y/2)+1;
	if (a == 9)
		return (selfn_*selfn_) + (y+1)/2 + ((x/2)-1)*selfn_;
}

bitset<128> walls_not_in_path(int x, int y, int *parent) {
	bitset<128> not_in_path;
	for (int i = 0 ; i < num_walls ; ++i)
		not_in_path.set(i);

	int x_, y_;
	int ax, ay;
	while (*(parent + ((x*selfn + y) * 2) + 0)!=-1) {
		x_ = *(parent + ((x*selfn + y) * 2) + 0); // parent[x][y][0];
		y_ = *(parent + ((x*selfn + y) * 2) + 1); // parent[x][y][1];

		if (y == y_) {
			ax = (x+x_)/2;
			ay = y;
			if (ay < selfn-1)
				not_in_path.set(index_of_action(8,ax,ay)-1,0);
			if (ay-2>=0)
				not_in_path.set(index_of_action(8,ax,ay-2)-1,0);
		}

		if (x == x_) {
			ay = (y+y_)/2;
			ax = x;
			if (ax>=2)
				not_in_path.set(index_of_action(9,ax,ay)-1,0);
			if (ax+2<=selfn-1)
				not_in_path.set(index_of_action(9,ax+2,ay)-1,0);
		}

		x = x_;
		y = y_;
	}

	return not_in_path;
}


void printParent(int * parent) {
	int counter = 0;
	for (int x = 0 ; x < 9 ; ++x) {
		for (int y = 0 ; y < 9 ; ++y) {
			counter+=2;
			cout << "(" << *(parent + ((x*selfn + y) * 2) + 0) << ", " << *(parent + ((x*selfn + y) * 2) + 1) << ")  ";
		}
		cout << endl;
	}
}

bitset<128> find_path_and_non_blocking_walls(int x, int y, int x_target, int wall, int* block_) {
	bitset<128> non_blocking_walls;	// res will be stored here
	int block[selfn][selfn];
	copy(block_, block_+(selfn*selfn), &block[0][0]);

	// place the wall at the block
	int wx = 0, wy = 0;
	if (wall < (num_walls/2)) {
		wx = (wall / selfn_)*2 + 1;
		wy = (wall % selfn_)*2;
		block[wx][wy+2] = 1;
	} else {
		wx = ((wall-(num_walls/2)) / selfn_)*2 + 2;
		wy = ((wall-(num_walls/2)) % selfn_)*2 + 1;
		block[wx-2][wy] = 1;
	}
	block[wx][wy] = 1;
	// ============================

	int parent[selfn][selfn][2] = {};
	parent[x][y][0] = -1;
	parent[x][y][1] = -1;

	queue <pair<int,int> > next;
	int visited[selfn][selfn] = {};
	add_next_visits(next, x, y, x_target, &block[0][0], &visited[0][0], &parent[0][0][0]);
	while (!next.empty()) {
		pair<int, int> tmp = next.front(); next.pop();
		if (tmp.first == x_target) // found a path
			return walls_not_in_path(tmp.first, tmp.second, &parent[0][0][0]);

		add_next_visits(next, tmp.first, tmp.second, x_target, &block[0][0], &visited[0][0], &parent[0][0][0]);
	}

	return non_blocking_walls;
}

PyObject* bitset_to_pylist(bitset<128> data) {

  PyObject* listObj = PyList_New( num_walls );
	if (!listObj) {
		cout << "ERROR HERE" << endl;
	 throw logic_error("Unable to allocate memory for Python list");
	}
	for (int i = 0; i < num_walls ; ++i) {
		PyObject *num = PyInt_FromLong(data[i]);
		if (!num) {
			cout << "ERROR DUO" << endl;
			Py_DECREF(listObj);
			throw logic_error("Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}

void printBits(bitset<128> data) {
	for (int i = 0 ; i < 32 ; ++i)
		cout << data[i];
	cout << endl;
}

static PyObject* legalWalls(PyObject* self, PyObject* args)
{
	PyObject *buffobj;
	Py_buffer view;

	//get the passed PyObject
	if (!PyArg_ParseTuple(args, "O", &buffobj)) return NULL;
	//get buffer info
	if (PyObject_GetBuffer(buffobj, &view, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) return NULL;

	int block[selfn][selfn] = {};
	int x = 0, y = 0, x_ = 0, y_ = 0;

	char *pointer = (char*)view.buf;
	for (int i = 0 ; i < 4 ; ++i) {
		for (int j = 0 ; j < selfn ; ++j) {
			for (int k = 0 ; k < selfn ; ++k) {
				if (i == 0 && (int)*pointer == 1) {
					x = j;
					y = k;
				}
				if (i == 1 && (int)*pointer == 1) {
					x_ = j;
					y_ = k;
				}

				if (i == 2)
					block[j][k] = (int)*pointer;
				else if (i == 3)
					block[j][k] += (int)*pointer;

				pointer++;
			}
		}
	}

	// ALL POSSIBLE WALLS BASED ON BLOCKS
	bitset<128> all_walls = possible_walls(&block[0][0]);
	bitset<128> legal_walls;
	bitset<128> tried_walls;
	int next_try = -1;
	while (++next_try < num_walls) {
		if (next_try < num_walls && (all_walls[next_try] == 0 || tried_walls[next_try] == 1))
			continue;

		if (next_try >= num_walls) break;
		legal_walls |= find_path_and_non_blocking_walls(x,y,0,next_try,&block[0][0]) & find_path_and_non_blocking_walls(x_,y_,selfn-1,next_try,&block[0][0]) & all_walls;
		tried_walls |= legal_walls;
	}

	Py_DECREF(buffobj);
	return bitset_to_pylist(legal_walls);
}


static PyMethodDef myMethods[] = {
	{ "setup", setup, METH_VARARGS, "..." },
	{ "legalWalls", legalWalls, METH_VARARGS, "..." },
	{ NULL, NULL, 0, NULL }
};

PyMODINIT_FUNC initpathFinder(void)
{
	PyObject *m = Py_InitModule3("pathFinder", myMethods, "...");
	if (m == NULL)
		return;
}
