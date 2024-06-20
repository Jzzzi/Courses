template <typename Elem, typename Key>
class PriorityQueue{
    // Elem is the type of the element to be stored in the queue
    // Key is the type of the key to be used for priority
    // If bigHeap is true, the queue will be a max heap
    // If bigHeap is false, the queue will be a min heap
    public:
        PriorityQueue(bool bigHeap = true);
        ~PriorityQueue();
        void enqueue(Elem e, Key k);
        Elem dequeue();
        Elem peek();
        bool isEmpty();

    private:
        int _size;
        int _capacity;
        Elem *_arr;
        Key *_key;
        bool _bigHeap;
        bool priorTo(Key k1, Key k2);
        void _shrink();
        void _expand();
};

template <typename Elem, typename Key>
PriorityQueue<Elem, Key>::PriorityQueue(bool bigHeap = true){
    this->_size = 0;
    this->_capacity = 10;
    this->_arr = new Elem[this->_capacity];
    this->_key = new Key[this->_capacity];
    this->_bigHeap = bigHeap;
}

template <typename Elem, typename Key>
PriorityQueue<Elem, Key>::~PriorityQueue(){
    delete[] this->_arr;
    delete[] this->_key;
}

template <typename Elem, typename Key>
void PriorityQueue<Elem, Key>::_shrink(){
    if(this->_size < this->_capacity / 4){
        this->_capacity /= 2;
        Elem *newArr = new Elem[this->_capacity];
        Key *newKey = new Key[this->_capacity];
        for(int i = 0; i < this->_size; i++){
            newArr[i] = this->_arr[i];
            newKey[i] = this->_key[i];
        }
        delete[] this->_arr;
        delete[] this->_key;
        this->_arr = newArr;
        this->_key = newKey;
    }
}

template <typename Elem, typename Key>
void PriorityQueue<Elem, Key>::_expand(){
    if(this->_size == this->_capacity){
        this->_capacity *= 2;
        Elem *newArr = new Elem[this->_capacity];
        Key *newKey = new Key[this->_capacity];
        for(int i = 0; i < this->_size; i++){
            newArr[i] = this->_arr[i];
            newKey[i] = this->_key[i];
        }
        delete[] this->_arr;
        delete[] this->_key;
        this->_arr = newArr;
        this->_key = newKey;
    }
}

template <typename Elem, typename Key>
bool PriorityQueue<Elem, Key>::priorTo(Key k1, Key k2){
    if (this->_bigHeap)
    return k1 > k2;
    else
    return k1 < k2;
}

template <typename Elem, typename Key>
void PriorityQueue<Elem, Key>::enqueue(Elem e, Key k){
    this->_expand();
    this->_arr[this->_size] = e;
    this->_key[this->_size] = k;
    this->_size++;
    // Bubble up
    int i = this->_size - 1;
    while(i > 0){
        int j = (i - 1) / 2;
        if(this->priorTo(this->_key[j], this->_key[i])){
            break;
        }
        Elem tempElem = this->_arr[i];
        Key tempKey = this->_key[i];
        this->_arr[i] = this->_arr[j];
        this->_key[i] = this->_key[j];
        this->_arr[j] = tempElem;
        this->_key[j] = tempKey;
        i = j;
    }
}

template <typename Elem, typename Key>
Elem PriorityQueue<Elem, Key>::dequeue(){
    if(this->_size == 0){
        throw "Queue is empty";
    }
    Elem ret = this->_arr[0];
    this->_size--;
    this->_arr[0] = this->_arr[this->_size];
    this->_key[0] = this->_key[this->_size];
    // Bubble down
    int i = 0;
    while(i < this->_size){
        int j = 2 * i + 1;
        if(j >= this->_size){
            break;
        }
        if(j + 1 < this->_size && this->priorTo(this->_key[j + 1], this->_key[j])){
            j++;
        }
        if(this->priorTo(this->_key[i], this->_key[j])){
            break;
        }
        Elem tempElem = this->_arr[i];
        Key tempKey = this->_key[i];
        this->_arr[i] = this->_arr[j];
        this->_key[i] = this->_key[j];
        this->_arr[j] = tempElem;
        this->_key[j] = tempKey;
        i = j;
    }
    this->_shrink();
    return ret;
}

template <typename Elem, typename Key>
Elem PriorityQueue<Elem, Key>::peek(){
    if(this->_size == 0){
        throw "Queue is empty";
    }
    return this->_arr[0];
}

template <typename Elem, typename Key>
bool PriorityQueue<Elem, Key>::isEmpty(){
    return this->_size == 0;
}