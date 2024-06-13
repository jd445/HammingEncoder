package SeqDTCode;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
public class Main {
	public static void main(String[] args) throws Exception{
		int maxL = 4;
		// maximum length
		int g = 1;
		// gap constraint
		double threshold=0.1;
		//maximum value of Gini index in one node
		int minNum=2;
		// minimum number of sequence in one node
		double minSplit=0.0;
		//minimum value of decreased impurity generated by segmentation	
		int depth=0;
		//maximum depth of the tree
		boolean pru=true;
		//determine whether to prune
		Main m=new Main();
		ReadFile r=new ReadFile();
		ArrayList<String> Tdata=new ArrayList<String>();
		//array to store acc
		float[] acc=new float[5];
		for(int i=1;i<=5;i++){
			Tdata=r.Read("./train_data_fold"+i+".txt");
			Node root = new Node();
			System.out.println("g:" + g + " maxL:" + maxL +" threshold:"+ threshold + " minNum:" +
			minNum + " depth:" + depth + " pru:" + pru);
			//create the decision tree
			// creat a array to store the number of node in each tree
			root=m.Train(Tdata, g, maxL, threshold, minNum, minSplit, depth, pru);

			//testing set
			Tdata=r.Read("./test_data_fold"+i+".txt");
			//test the decision tree constructed
			// float acc=m.Test(Tdata,g,root);
			acc[i-1]=m.Test(Tdata,g,root);
			System.out.println("acc:"+acc[i-1]);
		}
		// System.out.println("acc:"+acc);
		float sum=0;
		for(int i=0;i<5;i++){
			sum+=acc[i];
		}

        // Calculate the average
        double average = sum / acc.length;
		String filePath = args[0];
        // Write the average to the file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath, true))) {
            writer.write("Average accuracy: " + average + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        System.out.println("Average accuracy: " + average);
	}
	
	public Main(){
	}
	
	/**
	 * build a decision tree based on the training set
	 * @param train: training set
	 * @return root node of decision tree
	 * @throws Exception
	 */
	public Node Train(ArrayList<String> train, int g, int maxL, double threshold,
			int minNum, double minSplit, int depth, boolean pru) throws Exception{
		
		Tree tree = new Tree(g,maxL,threshold,minNum,minSplit,depth,pru);
		
		Node root = new Node();
		root = tree.createRoot(train);
		return root;
	}
	
	/**
	 * use a decision tree to classify sequences
	 * @param test: the testing set
	 * @param root: the root node of the decision tree
	 * @return classification accuracy
	 */
	public float Test(ArrayList<String> test, int g, Node root){
		Classification cf = new Classification(g);
		int correct = 0;
		int all=0;
		for(String str:test){	
			all++;
			String[] s = str.split("\t");
			String label=s[0];
			String[] dataInstanceForTesting=s[1].split(" ");
			String TempLabel=cf.getLabel(root, dataInstanceForTesting);
			if(TempLabel.equals(label)){
				correct++;
			}else{
				//System.out.println(str);
			}
		}
		return (float)correct/all;
	}
}
