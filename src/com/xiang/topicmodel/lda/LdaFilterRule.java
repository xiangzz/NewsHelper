/**
 * 
 */
package com.xiang.topicmodel.lda;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.StringTokenizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import liuyang.nlp.lda.com.FileUtil;

import com.xiang.topicmodel.framework.FilterRule;

/**
 * @author rivercrab3
 *
 */
public class LdaFilterRule extends FilterRule {
	private String stopwordFile;
	private HashSet<String> stopwords;
	
	
	private ArrayList<String> readFile(File file){
		BufferedReader reader = null;
		ArrayList<String> words = new ArrayList<String>();
		try {
			reader = new BufferedReader(new FileReader(file));
			String line = null;
			while ((line = reader.readLine()) != null) {
				StringTokenizer strTok = new StringTokenizer(line);
				while (strTok.hasMoreTokens()) {
					String token = strTok.nextToken();
					words.add(token.toLowerCase().trim());
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return words;
	}
	
	public LdaFilterRule(String fileName){
		File file = new File(fileName);
		ArrayList<String> words = this.readFile(file);
		this.stopwords = new HashSet<String>();
		this.stopwords.addAll(words);
	}
	
    
	@Override
	public Boolean isFit(String word) {
		// TODO Auto-generated method stub
		Boolean fit = true;
		if(stopwords.contains(word.toLowerCase()) || isNoiseWord(word)){
			fit = false;
		}
		return fit;
	}
	
	public boolean isNoiseWord(String string) {
		// TODO Auto-generated method stub
		// filter @xxx and URL
		if(string.matches(".*www\\..*") || string.matches(".*\\.com.*") || 
				string.matches(".*http:.*") ){
			return true;
		}

		Pattern MY_PATTERN = Pattern.compile("[a-zA-Z\u4e00-\u9fa5]+");
		Matcher m = MY_PATTERN.matcher(string);
		if (!m.matches()) {
			return true;
		} else {
			return false;
		}
	}
}
