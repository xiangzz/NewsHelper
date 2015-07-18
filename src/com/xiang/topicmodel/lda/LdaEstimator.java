/**
 * 
 */
package com.xiang.topicmodel.lda;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import com.xiang.topicmodel.lda.FileUtil;


import com.xiang.topicmodel.framework.Estimator;

/**
 * @author rivercrab3
 *
 */
public class LdaEstimator extends Estimator {
	
	float alpha; //doc-topic dirichlet prior parameter 
	float beta; //topic-word dirichlet prior parameter
	int V, K, M;//vocabulary size, topic number, document number
	String paramFileName;

	int iterations;//Times of iterations
	int saveStep;//The number of iterations between two saving
	int beginSaveIters;//Begin save model at this iteration
    
	
	public LdaEstimator(String paramFileName) throws IOException{
		this.paramFileName = paramFileName;

		Properties prop = new Properties();
		File pFile = new File(paramFileName);
		FileInputStream pfin = new FileInputStream(pFile);
		prop.load(pfin);
		this.alpha = Float.parseFloat(prop.getProperty(("alpha")));
		this.beta = Float.parseFloat(prop.getProperty(("beta")));
		this.iterations = Integer.parseInt(prop.getProperty(("iterations")));
		this.K = Integer.parseInt(prop.getProperty(("K")));
		this.saveStep = Integer.parseInt(prop.getProperty(("saveStep")));
		this.beginSaveIters = Integer.parseInt(prop.getProperty(("beginSaveIters")));
	}
	
	
	/**
	 * ��������δ�����ĵ����в���
	 * @param n
	 * @param d_nk
	 * @param d_nkSum
	 * @param d_nkt
	 * @param d_nktSum
	 * @param phi
	 * @return
	 */
	private int sampleTopicZ(int n,int[] d_z,int[] d_doc,Map<Integer,Integer> transTable,
			int[] d_nk, Integer d_nkSum,
			int[][] d_nkt, int[] d_nktSum,
			Double[][] phi){
		
		int d_V = d_nkt[0].length;
		int oldTopic = d_z[n];
		int origin_t = d_doc[n];
		int t = transTable.get(origin_t);
		
		d_nk[oldTopic]--;
		d_nkt[oldTopic][t]--;
		d_nkSum--;
		d_nktSum[oldTopic]--;
		//���� p(z_i = k|z_-i, w)
		double [] p = new double[K];
		for(int k = 0; k < K; k++){
			p[k] = (d_nkt[k][t] + beta) / (d_nktSum[k] + d_V * beta) * phi[k][t];
		}
		//���ö��ַ�ȷ��topic
		for(int k = 1; k < K; k++){
			p[k] += p[k - 1];
		}
		double u = Math.random() * p[K - 1]; //p[] is unnormalised
		int newTopic;
		for(newTopic = 0; newTopic < K; newTopic++){
			if(u < p[newTopic]){
				break;
			}
		}
		//��������ĵ���n���ʵ�topic
		d_nk[newTopic]++;
		d_nkt[newTopic][t]++;
		d_nkSum++;
		d_nktSum[newTopic]++;
		return newTopic;
		
	}
	
	private double[] classify(LdaDataProc ldp , Map<String, Integer> termToIndexMap,Double[][] phi){
		ArrayList<ArrayList<String>> docs = ldp.docs;
		if(docs.size() > 1 || docs.size() < 1){
			return null;
		}
		ArrayList<String> words = docs.get(0);
		ArrayList<String> oldWords = words;
		ArrayList<String> newWords = new ArrayList<String>();
		Set<String> wordSet = new HashSet<String>();
		Map<Integer,Integer> transTable = new HashMap<Integer,Integer>();
	
		//��ѵ��������û�е�word�ų�����ͨ������ȥ��
		for(String word : oldWords){
			if(termToIndexMap.containsKey(word)){
				newWords.add(word);
				wordSet.add(word);
			}
		}
		
		//�������
		int nWords = newWords.size();
		int d_V = wordSet.size();
	
		//d_thetaΪ����ֲ�
		double[] d_theta = new double[K];
		//d_z��¼ÿ��word������
		int[] d_z = new int[nWords];
		//d_V��ʾ���ظ���word�ĸ���
		
		//d_nk��ʾ              topic1  topic2  ...  topicK
		//      doc  0.1     0.1     ...  0.1 
		int[] d_nk = new int[K];
		//d_nkSum��ʾd_nk��ÿ�е�sum
		int d_nkSum = 0;
		
		int[][] d_nkt = new int[K][d_V];
		int[] d_nktSum = new int[K];
		
		
		//d_doc��ʾÿ���ʵ����
		int[] d_doc = new int[nWords];
		
		//��doc�е�word���
		int itemNum = 0;
		for(int i = 0; i < nWords; ++i){
			String word = newWords.get(i);
			int index = termToIndexMap.get(word);
			d_doc[i] = index;
			if (!transTable.containsKey(index)){
				transTable.put(index, itemNum++);
			}
		}
		
		//��ʼ��topic
		for(int i = 0; i < nWords; ++i){
			int initTopic = (int)(Math.random() * K);
			d_z[i] = initTopic;
			//number of words in doc m assigned to topic initTopic add 1
			d_nk[initTopic]++;
			//number of terms doc[m][n] assigned to topic initTopic add 1
			int t = d_doc[i];
			int trans_t = transTable.get(t);
			d_nkt[initTopic][trans_t]++;
			// total number of words assigned to topic initTopic add 1
			d_nktSum[initTopic]++;
		}
		d_nkSum = nWords;
		
		//��ʼ����d_theta
		int d_iterations = 100;
		int sample_step = 5;
		for(int i = 0; i < d_iterations; i++){
			
			//�Բ����������в�������
			if(i % d_iterations == sample_step){
				for(int k = 0; k < K; k++){
					d_theta[k] = (d_nk[k] + alpha) / (d_nkSum + K * alpha);
				}
			}
			
			//�ظ����м���˹����,����d_z
			for(int n = 0; n < nWords; n++){
				// Sample from p(z_i|z_-i, w)
				//�������ö�̬���ٶ���һ��sampleTopicZ����
				int newTopic = sampleTopicZ(n,d_z,d_doc,transTable,
						d_nk,d_nkSum,d_nkt,d_nktSum,phi);
				d_z[n] = newTopic;
			}
		}
		
		//��һ������
		double d_thetaSum = 0.0;
		for (double x:d_theta){
			d_thetaSum += x;
		}
		for (int i=0;i< d_theta.length;++i){
			d_theta[i] = d_theta[i] / d_thetaSum;
		}
		return d_theta;
		
	}
	
	public double[] estimate(LdaDataProc newDoc) throws IOException{

		//��ȡphi��termToIndexMap
		Map<String,Integer> termToIndexMap = FileUtil.readMapfromFile(this.paramFileName, "termToIndexMap");
		Double[][] phi = FileUtil.readMatrixFromFile(this.paramFileName);
		
		//��ʼ��Model

		//��ʼ����
		double[] res = this.classify(newDoc, termToIndexMap, phi);
		for (int i = 0;i<res.length;++i){
			System.out.println("topic"+i+":"+res[i]);
		}
		return res;
		
	}
}
