package com.xiang.topicmodel.test.lda;

import java.io.IOException;



import com.xiang.topicmodel.lda.LdaDataProc;
import com.xiang.topicmodel.lda.LdaEstimator;
import com.xiang.topicmodel.lda.LdaFilter;
import com.xiang.topicmodel.lda.LdaFilterRule;
import com.xiang.topicmodel.lda.LdaTrainer;

public class LdaTrainerTest {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		LdaFilterRule rule = new LdaFilterRule("./data/stopWords");
		LdaFilter filter = new LdaFilter(rule);
		/*
		LdaDataProc dp = new LdaDataProc("./data/news", filter);
		dp.read();
		dp.process();
		*/
		
		/*
		ArrayList< ArrayList<String> > d = (ArrayList< ArrayList<String> >) dp.getData();
		for(ArrayList<String> al:d){
			System.out.println("-------------------");
			for(String s:al){
				System.out.print(s+",");
			}
			System.out.println();
		}
		*/
		/*
		LdaTrainer tra = new LdaTrainer("./data/parameters.properties", dp);
		tra.train();
		*/
		
		LdaDataProc newDocProc = new LdaDataProc("./data/news/news1", filter);
		newDocProc.read();
		newDocProc.process();
		LdaEstimator es = new LdaEstimator("./data/parameters.properties");
		es.estimate(newDocProc);
	}

}
