package com.ljh.bnndemo;

import java.util.Arrays;
import java.util.HashMap;

public class Utils {
    public int[] getTopNFromArray(float[] arr,int len,int N){
        HashMap<Float,Integer> hashMap = new HashMap<Float,Integer>();
        for(int i=0;i<len;i++){
            hashMap.put(arr[i],i);
            arr[i] = arr[i] *(-1);
        }
        Arrays.sort(arr);
        int[] index = new int[N];
        for(int i=0;i<N;i++){
            index[i] = hashMap.get(arr[i]*(-1));
        }
        return index;
    }
}
