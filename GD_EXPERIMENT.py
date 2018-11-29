import tensorflow as tf
import numpy
def IFF_GD_deep_learninig(num_of_exper, learnRate, nb_hidden, shortCut):
 dim=2
 nb_outputs = 1
 max_epocs = 40000
 temperature = 1
 EXCEPT_VALID_CHANGE=0.0001
 EXCEPT_VALID_VALUE=0.2
 NUMBER_OF_SUCESS_EXPERIMENT=10
 if shortCut:
     nb_hbridge = nb_hidden + dim  # Bridge inputs to output (highway)
 else:
     nb_hbridge = nb_hidden


 x = tf.placeholder(tf.float32, [None, dim])      # define input placeholders and variables
 t = tf.placeholder(tf.float32,[None,1])
 w1 = tf.Variable(tf.random_uniform([dim, nb_hidden], -1, 1,seed=0)) # random weights
 w2 = tf.Variable(tf.random_uniform([nb_hbridge, nb_outputs], -1, 1,seed=0))
 b1 = tf.Variable(tf.zeros([nb_hidden]))   # biases are zeros (not random)
 b2 = tf.Variable(tf.zeros([nb_outputs]))
 z1=tf.matmul(x, w1) + b1
 hlayer1 = tf.sigmoid(z1/temperature)
 if shortCut :  hlayer1=tf.concat([hlayer1, x],1)
 z2=tf.matmul(hlayer1, w2) + b2
 out = tf.sigmoid(z2/temperature)
 loss = -tf.reduce_sum(t*tf.log(out)+(1-t)*tf.log(1-out)) # Xross Entropy Loss
 optimizer = tf.train.GradientDescentOptimizer(learnRate) #Grad Descent Optimizer
 train = optimizer.minimize(loss)  # training is running optimizer to minimize loss

 x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
 t_train = [[1], [0], [0], [1]]
 x_valid = [[0, 0], [0, 1], [1, 0], [1, 1], [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]];
 t_valid = [[1], [0], [0], [1], [1], [0], [0], [1]]
 init = tf.global_variables_initializer()
 sess = tf.Session()
 sess.run(init)
 sucess_Exper=0
 all_Epocs=[]
 all_loss_Valid=[]
 all_loss_Train=[]
 num_of_Failures=0;
 while(sucess_Exper <10):
  last_validation_loss = 0
  improve_counter = 0
  experiment_Sucess = False
  for i in range(max_epocs):
     curr_train, curr_loss_train = sess.run([train, loss],{x: x_train, t: t_train})
     curr_out, valid_loss = sess.run([out, loss], {x: x_valid, t: t_valid})
     if(last_validation_loss-valid_loss<EXCEPT_VALID_CHANGE ):
         if(improve_counter<NUMBER_OF_SUCESS_EXPERIMENT):
            improve_counter+=1
     else:
       improve_counter=0
     last_validation_loss=valid_loss
     if (improve_counter == NUMBER_OF_SUCESS_EXPERIMENT and last_validation_loss < EXCEPT_VALID_VALUE):
         experiment_Sucess=True
         all_Epocs.append(i)
         all_loss_Train.append(curr_loss_train)
         all_loss_Valid.append(last_validation_loss)
         break
  if experiment_Sucess:
      sucess_Exper+=1

  else:
      num_of_Failures+=1
  W1 = w1.assign(tf.random_uniform([dim, nb_hidden], -1, 1, seed=sucess_Exper*10+num_of_Failures*100))  # random weights
  W2 = w2.assign(tf.random_uniform([nb_hbridge, nb_outputs], -1, 1, seed=sucess_Exper*10+num_of_Failures*100))
  B1 = b1.assign(tf.zeros([nb_hidden]))  # biases are zeros
  B2 = b2.assign(tf.zeros([nb_outputs]))
  sess.run([W1, W2, B1, B2])
 print("experiment",num_of_exper ,"hidden:",nb_hidden,"LR",learnRate,"Bridge",shortCut,"Failures:",num_of_Failures)
 print("meanepocs:",sum(all_Epocs)/10,"std/epocs%",numpy.std(all_Epocs,ddof=1))
 print("meanvalidloss:",sum(all_loss_Valid)/10,"stdvalidlossPercent:",numpy.std(all_loss_Valid,ddof=1))
 print("meanTrainLoss:",sum(all_loss_Train)/10,"stdTrainLossPercent:",numpy.std(all_loss_Train,ddof=1))


########################################################################
IFF_GD_deep_learninig(1, 0.1, 4, True)
IFF_GD_deep_learninig(2, 0.1, 4, False)
IFF_GD_deep_learninig(3, 0.1, 2, True)
IFF_GD_deep_learninig(4, 0.1, 2, False)
IFF_GD_deep_learninig(5, 0.01, 4, True)
IFF_GD_deep_learninig(6, 0.01, 4, False)
IFF_GD_deep_learninig(7, 0.01, 2, True)
IFF_GD_deep_learninig(8, 0.01, 2, False)