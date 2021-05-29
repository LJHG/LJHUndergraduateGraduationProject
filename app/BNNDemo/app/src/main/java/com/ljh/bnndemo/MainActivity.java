package com.ljh.bnndemo;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    private static final int SELECT_IMAGE = 1;

    private List<String> words;
    private Button button1;
    private ImageView imageView;
    private TextView textView;
    private Bitmap yourSelectedImage = null;
    private Net net = null;
    private Utils utils = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        words = new ArrayList<String>();
        //读取label来初始化
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    getAssets().open("labels.txt")
            ));
            while (true) {
                String line = reader.readLine();
                if (line == null) break;
                words.add(line);
            }

        } catch (IOException e) {
            System.out.println("error here");
            e.printStackTrace();
        }



        button1 = (Button)findViewById(R.id.button);
        imageView = (ImageView)findViewById(R.id.imageView);
        textView = findViewById(R.id.sample_text);

        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });


        utils = new Utils();

        // Example of a call to a native method
        //tv.setText(testForJNI());
        //Student stu  = new Student();
        //stu.setValue("yoyochecknow");
        //tv.setText(stu.getValue());
        net = new Net();
//        net.initialize();
//        net.predict();
        textView.setText("选择图片来进行识别吧!");

        //load model sort of things(solution from https://stackoverflow.com/questions/65273837/android-native-file-read-from-assets-folder-by-tflite-buildfromfile)
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = getResources().getAssets().openFd("bireal_best.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer modelBuffer = null;
        try {
            modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } catch (IOException e) {
            e.printStackTrace();
        }

        //load model
        net.loadModel(modelBuffer);

//        //fill in input
//        float input[]  = new float[150528];
//        for(int i=0;i<50176;i++)
//        {
//            input[i*3 + 0] = 1;
//            input[i*3 + 1] = 2;
//            input[i*3 + 2] = 3;
//        }
//        //predict
//        long start=System.currentTimeMillis();
//        float[] output = net.predict(input);
//        long end = System.currentTimeMillis();
//        System.out.println("Inference time: "+(end-start)+" ms");
//
//        //present output
//        System.out.println("print top 5");
//        int[] indexs = utils.getTopNFromArray(output,1000,5);
//        for(int i=0;i<5;i++){
//            System.out.println(indexs[i]);
//        }
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            System.out.println("here is uri");
            System.out.println("******************************");
            System.out.println(selectedImage);
            System.out.println("******************************");

            try
            {
                if (requestCode == SELECT_IMAGE) {
                    Bitmap bitmap = decodeUri(selectedImage);

                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                    // resize to 227x227
                    yourSelectedImage = Bitmap.createScaledBitmap(rgba, 32, 32, false);

                    rgba.recycle();

                    imageView.setImageBitmap(bitmap);

                }
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }


            //start to predict
            byte[] bgrByteArr = bitmap2BGR(yourSelectedImage);

            float[] floatData = new float[bgrByteArr.length];
            int [] intData = new int[bgrByteArr.length];
            System.out.println(bgrByteArr.length);
            for(int j=0;j<bgrByteArr.length;j++){
                intData[j] = Byte.toUnsignedInt(bgrByteArr[j]);
                floatData[j] = intData[j];
            }

            //the length of flaot data is 3072
            //z-score
            float mean = (float) 121.93584;
            float std = (float) 68.38902;

            for(int i=0;i<3072;i++){
                floatData[i] = (float) ((floatData[i] - mean)/ (std +1e-7));
            }



            //这里应该还是有离谱，不然每次都要重新读模型？那也太扯了
            //load model sort of things(solution from https://stackoverflow.com/questions/65273837/android-native-file-read-from-assets-folder-by-tflite-buildfromfile)
            AssetFileDescriptor fileDescriptor = null;
            try {
                fileDescriptor = getResources().getAssets().openFd("bireal_best.tflite");
            } catch (IOException e) {
                e.printStackTrace();
            }
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer = null;
            try {
                modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            } catch (IOException e) {
                e.printStackTrace();
            }


            //load model
            net.loadModel(modelBuffer);

            long startTime = System.currentTimeMillis();    //获取开始时间
            float[] output = net.predict(floatData);
            long endTime = System.currentTimeMillis();
            System.out.println("print top 5");
            int[] indexs = utils.getTopNFromArray(output,100,5);
            for(int i=0;i<5;i++){
                System.out.println(indexs[i]);
            }
            String ans = "";
            ans += "类别：" + words.get(indexs[0]) + "\n";
            ans += "推理时间：" + (endTime-startTime) +"  ms  \n";
            textView.setText(ans);

        }


    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 400;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                    || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);
    }

    // 把bitmap转为byte数组的函数
    public static byte[] bitmap2BGR(Bitmap bitmap)
    {
        int bytes = bitmap.getByteCount();

        ByteBuffer buffer = ByteBuffer.allocate(bytes);
        bitmap.copyPixelsToBuffer(buffer);

        byte[] rgba = buffer.array();
        byte[] pixels = new byte[(rgba.length / 4) * 3];

        int count = rgba.length / 4;

        for (int i = 0; i < count; i++) {
            pixels[i * 3] = rgba[i * 4 + 2];      //B
            pixels[i * 3 + 1] = rgba[i * 4 + 1];  //G
            pixels[i * 3 + 2] = rgba[i * 4];      //R
        }
        return pixels;
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}