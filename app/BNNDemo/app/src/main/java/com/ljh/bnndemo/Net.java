package com.ljh.bnndemo;

import java.nio.ByteBuffer;

public class Net {
    static {
        System.loadLibrary("lce");
    }

    public native String testForJNI();
    public native boolean loadModel(ByteBuffer modelBuffer);
    public native float[] predict(float[] input);
}

